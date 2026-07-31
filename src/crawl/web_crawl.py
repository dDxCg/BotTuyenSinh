import asyncio
import io
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pypdf
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

sources = [
    "https://vinuni.edu.vn/vi/vingroup-tang-toc-dao-tao-20-000-nhan-tai-ai-thuc-chien/",
    "https://vinuni.edu.vn/vi/thong-tin-tuyen-sinh-chuong-trinh-dao-tao-nhan-tai-ai-thuc-chien-khoa-co-ban/",
    "https://vinuni.edu.vn/wp-content/uploads/2025/04/20K-AI-Handbook_final.pdf"
]

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "web"
CLEAN_DIR = OUT_DIR / "_clean"
RAW_DIR = OUT_DIR / "_raw"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)

# Bỏ ở tầng HTML: mega-menu, lang switcher, sidebar, footer, script/style.
EXCLUDED_TAGS = [
    "nav", "header", "footer", "aside", "form",
    "script", "style", "noscript", "svg", "iframe",
]

# vinuni.edu.vn có Cloudflare: UA mặc định của crawl4ai (Linux/Chrome 116) bị chặn
# từ request thứ 2. Cần stealth + UA thật + simulate_user.
BROWSER_CONFIG = BrowserConfig(headless=True, enable_stealth=True, user_agent=USER_AGENT)

HTML_CONFIG = CrawlerRunConfig(
    excluded_tags=EXCLUDED_TAGS,
    cache_mode=CacheMode.BYPASS,
    simulate_user=True,
    magic=True,
    # networkidle timeout vì trang có analytics polling — dùng domcontentloaded + chờ cố định.
    wait_until="domcontentloaded",
    delay_before_return_html=3.0,
    verbose=False,
)

DELAY_SECONDS = 5
MIN_CHARS = 1000

IMAGE_ONLY = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
LINK_ONLY = re.compile(r"^\s*(?:!?\[[^\]]*\]\([^)]*\)\s*)+$")


def slugify(url: str) -> str:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    name = parts[-1] if parts else parsed.netloc
    name = re.sub(r"\.(pdf|html?|php|aspx)$", "", name, flags=re.I)
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "page"


def clean_markdown(md: str) -> str:
    """Bỏ phần rác còn sót sau khi excluded_tags đã cắt nav/footer."""
    kept = []
    for line in md.splitlines():
        stripped = line.strip()
        if IMAGE_ONLY.match(line):
            continue
        if stripped.startswith("[Trang chủ]("):
            continue
        if stripped == "Mục lục":
            continue
        if stripped and LINK_ONLY.match(line):
            continue
        kept.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def despace(text: str) -> str:
    """PDF thiết kế có letter-tracking: pypdf trả về từng ký tự cách nhau 1 space,
    ranh giới từ là 2 space. Gộp lại theo quy ước đó."""
    out = []
    for line in text.splitlines():
        tokens = [t for t in line.split(" ") if t]
        if tokens and sum(1 for t in tokens if len(t) == 1) / len(tokens) > 0.6:
            line = line.replace("  ", "\x00").replace(" ", "").replace("\x00", " ")
        out.append(line.rstrip())
    return "\n".join(out)


def pdf_to_markdown(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": f"{urlparse(url).scheme}://{urlparse(url).netloc}/",
            "Accept": "application/pdf,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        body = despace(page.extract_text() or "").strip()
        if body:
            pages.append(f"## Trang {i}\n\n{body}")
    return "\n\n".join(pages)


def save(url: str, raw: str) -> None:
    slug = slugify(url)
    if len(raw) < MIN_CHARS:
        print(f"WARN {slug}: chỉ {len(raw)} chars — nghi bị chặn, không ghi file")
        return
    cleaned = clean_markdown(raw)
    header = f"<!-- source: {url} -->\n\n"
    (RAW_DIR / f"{slug}.md").write_text(header + raw, encoding="utf-8")
    (CLEAN_DIR / f"{slug}.md").write_text(header + cleaned, encoding="utf-8")
    drop = 100 - round(100 * len(cleaned) / len(raw))
    print(f"saved {slug}.md raw={len(raw)} clean={len(cleaned)} (-{drop}%)")


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    pdf_urls = [u for u in sources if urlparse(u).path.lower().endswith(".pdf")]
    html_urls = [u for u in sources if u not in pdf_urls]

    # Mỗi URL một browser session riêng: dùng lại session cho request thứ 2 thì
    # Cloudflare bắn JS challenge.
    for i, url in enumerate(html_urls):
        if i:
            await asyncio.sleep(DELAY_SECONDS)
        async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
            result = await crawler.arun(url=url, config=HTML_CONFIG)
        if not result.success:
            print(f"WARN {slugify(url)}: crawl failed — {result.error_message}")
            continue
        save(url, result.markdown.raw_markdown or "")

    for url in pdf_urls:
        try:
            save(url, pdf_to_markdown(url))
        except Exception as exc:
            print(f"WARN {slugify(url)}: pdf failed — {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
