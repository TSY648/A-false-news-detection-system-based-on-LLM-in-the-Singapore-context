import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET
from urllib.parse import urljoin, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CSV_FIELDS = ["source_id", "title", "content", "source_type", "url", "published_at"]


@dataclass
class CrawlTarget:
    source_id: str
    url: str
    source_type: str
    title_hint: str = ""


@dataclass
class DiscoverySeed:
    seed_url: str
    source_type: str
    source_id_prefix: str
    allowed_path_prefixes: Tuple[str, ...]
    denied_path_prefixes: Tuple[str, ...] = ()
    sitemap_urls: Tuple[str, ...] = ()
    title_hint: str = ""
    max_links: int = 300


DEFAULT_TARGETS: List[CrawlTarget] = [
    # POFMA Office historical reports / media releases
    CrawlTarget(
        source_id="pofma-2020-001",
        url="https://www.pofmaoffice.gov.sg/files/documents/media-releases/2020/january/pofma-office-gcd-media-statement-27-jan-final.pdf",
        source_type="pofma_report",
        title_hint="POFMA media release on false COVID-19 death claim on HardwareZone",
    ),
    CrawlTarget(
        source_id="pofma-2022-001",
        url="https://www.pofmaoffice.gov.sg/files/media-releases/pofma-pr-ind-12feb2022-01.pdf",
        source_type="pofma_report",
        title_hint="POFMA press release on Wake Up, Singapore falsehoods",
    ),
    CrawlTarget(
        source_id="pofma-2022-002",
        url="https://www.pofmaoffice.gov.sg/files/media-releases/pofma-pr-pofma-31mar2022-01.pdf",
        source_type="pofma_report",
        title_hint="POFMA warning on preschooler COVID-19 falsehood",
    ),
    # Factually articles
    CrawlTarget(
        source_id="factually-2020-001",
        url="https://www.factually.gov.sg/corrections-and-clarifications/factually-clarification-on-falsehood-posted-by-sst-on-quarantine-of-foreign-workers/",
        source_type="factually_article",
    ),
    CrawlTarget(
        source_id="factually-2020-002",
        url="https://www.factually.gov.sg/corrections-and-clarifications/factually-clarifications-on-falsehoods-posted-by-str-on-covid-19-situation/",
        source_type="factually_article",
    ),
    CrawlTarget(
        source_id="factually-2021-001",
        url="https://www.factually.gov.sg/corrections-and-clarifications/factually150821/",
        source_type="factually_article",
    ),
    CrawlTarget(
        source_id="factually-2022-001",
        url="https://www.factually.gov.sg/corrections-and-clarifications/inaccurate-statements-about-the-labour-market",
        source_type="factually_article",
    ),
    CrawlTarget(
        source_id="factually-2023-001",
        url="https://www.factually.gov.sg/corrections-and-clarifications/factually240823-a/",
        source_type="factually_article",
    ),
    CrawlTarget(
        source_id="factually-2024-001",
        url="https://www.factually.gov.sg/corrections-and-clarifications/factually220224",
        source_type="factually_article",
    ),
    CrawlTarget(
        source_id="factually-2025-001",
        url="https://www.factually.gov.sg/corrections-and-clarifications/how-cpf-monies-are-invested/",
        source_type="factually_article",
    ),
    # FAQ pages
    CrawlTarget(
        source_id="moh-faq-2016-001",
        url="https://www.moh.gov.sg/newsroom/faq-impact-of-haze-on-health/",
        source_type="faq_moh",
    ),
    CrawlTarget(
        source_id="moh-faq-2016-002",
        url="https://www.moh.gov.sg/newsroom/faq-use-of-masks-and-availability-of-masks/",
        source_type="faq_moh",
    ),
    CrawlTarget(
        source_id="mom-faq-2024-001",
        url="https://www.mom.gov.sg/faq/work-pass-general/can-a-work-pass-holder-work-in-multiple-jobs",
        source_type="faq_mom",
    ),
    CrawlTarget(
        source_id="mom-faq-2025-001",
        url="https://www.mom.gov.sg/faq/work-pass-general/how-do-i-check-if-a-work-pass-is-valid",
        source_type="faq_mom",
    ),
    CrawlTarget(
        source_id="hdb-faq-2025-001",
        url="https://www.hdb.gov.sg/e-resale/faq",
        source_type="faq_hdb",
    ),
    CrawlTarget(
        source_id="hdb-faq-2025-002",
        url="https://www.hdb.gov.sg/buying-a-flat",
        source_type="faq_hdb",
    ),
    CrawlTarget(
        source_id="cpf-faq-2024-001",
        url="https://www.cpf.gov.sg/service/article/how-do-i-contact-the-cpf-board",
        source_type="faq_cpf",
    ),
    CrawlTarget(
        source_id="cpf-faq-2025-001",
        url="https://www.cpf.gov.sg/service/article/when-can-i-withdraw-my-cpf-savings",
        source_type="faq_cpf",
    ),
    CrawlTarget(
        source_id="cpf-faq-2026-001",
        url="https://www.cpf.gov.sg/service/article/how-do-i-inform-cpf-board-if-i-have-changed-my-contact-details",
        source_type="faq_cpf",
    ),
]


DEFAULT_DISCOVERY_SEEDS: List[DiscoverySeed] = [
    DiscoverySeed(
        seed_url="https://www.factually.gov.sg/corrections-and-clarifications/",
        source_type="factually_article",
        source_id_prefix="factually-auto",
        allowed_path_prefixes=("/corrections-and-clarifications/",),
        denied_path_prefixes=("/corrections-and-clarifications", "/corrections-and-clarifications/"),
        sitemap_urls=("https://www.factually.gov.sg/sitemap.xml",),
        max_links=400,
    ),
    DiscoverySeed(
        seed_url="https://www.pofmaoffice.gov.sg/",
        source_type="pofma_report",
        source_id_prefix="pofma-root-auto",
        allowed_path_prefixes=(
            "/media-centre/press-releases/file-",
            "/files/media-releases/",
            "/files/documents/media-releases/",
        ),
        sitemap_urls=("https://www.pofmaoffice.gov.sg/sitemap.xml",),
        max_links=200,
    ),
    DiscoverySeed(
        seed_url="https://www.pofmaoffice.gov.sg/media-centre/press-releases/",
        source_type="pofma_report",
        source_id_prefix="pofma-auto",
        allowed_path_prefixes=(
            "/media-centre/press-releases/file-",
            "/files/media-releases/",
            "/files/documents/media-releases/",
        ),
        sitemap_urls=("https://www.pofmaoffice.gov.sg/sitemap.xml",),
        max_links=200,
    ),
    DiscoverySeed(
        seed_url="https://www.mom.gov.sg/faq",
        source_type="faq_mom",
        source_id_prefix="mom-faq-auto",
        allowed_path_prefixes=("/faq/",),
        denied_path_prefixes=("/faq",),
        sitemap_urls=("https://www.mom.gov.sg/sitemap.xml",),
        max_links=300,
    ),
    DiscoverySeed(
        seed_url="https://www.moh.gov.sg/",
        source_type="faq_moh",
        source_id_prefix="moh-root-auto",
        allowed_path_prefixes=("/newsroom/",),
        sitemap_urls=("https://www.moh.gov.sg/sitemap.xml",),
        max_links=500,
    ),
    DiscoverySeed(
        seed_url="https://www.moh.gov.sg/newsroom/faq-impact-of-haze-on-health/",
        source_type="faq_moh",
        source_id_prefix="moh-faq-auto",
        allowed_path_prefixes=("/newsroom/",),
        sitemap_urls=("https://www.moh.gov.sg/sitemap.xml",),
        max_links=500,
    ),
    DiscoverySeed(
        seed_url="https://www.cpf.gov.sg/",
        source_type="faq_cpf",
        source_id_prefix="cpf-root-auto",
        allowed_path_prefixes=("/service/article/",),
        sitemap_urls=("https://www.cpf.gov.sg/sitemap.xml",),
        max_links=500,
    ),
    DiscoverySeed(
        seed_url="https://www.cpf.gov.sg/service/article/when-can-i-withdraw-my-cpf-savings",
        source_type="faq_cpf",
        source_id_prefix="cpf-faq-auto",
        allowed_path_prefixes=("/service/article/",),
        sitemap_urls=("https://www.cpf.gov.sg/sitemap.xml",),
        max_links=500,
    ),
    DiscoverySeed(
        seed_url="https://www.hdb.gov.sg/e-resale/faq",
        source_type="faq_hdb",
        source_id_prefix="hdb-faq-auto",
        allowed_path_prefixes=("/e-resale/faq", "/buying-a-flat", "/residential/buying-a-flat"),
        sitemap_urls=("https://www.hdb.gov.sg/sitemap.xml",),
        max_links=120,
    ),
]


class MinimalContentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_tags = {"script", "style", "noscript", "svg"}
        self._skip_depth = 0
        self._in_title = False
        self._heading_tag = None
        self.title_parts: List[str] = []
        self.heading_parts: List[str] = []
        self.text_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth > 0:
            return
        if tag == "title":
            self._in_title = True
        if tag in {"h1", "h2"} and self._heading_tag is None:
            self._heading_tag = tag
        if tag in {"p", "li", "br", "div", "section", "article", "main", "h1", "h2", "h3"}:
            self.text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if self._heading_tag == tag:
            self._heading_tag = None
        if tag in {"p", "li", "div", "section", "article", "main"}:
            self.text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        cleaned = normalize_text(data)
        if not cleaned:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
        if self._heading_tag is not None:
            self.heading_parts.append(cleaned)
        self.text_parts.append(cleaned)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._current_href: Optional[str] = None
        self._current_text: List[str] = []
        self.links: List[Tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        attr_map = {str(k).lower(): str(v) for k, v in attrs}
        href = (attr_map.get("href") or "").strip()
        if not href:
            return
        self._current_href = href
        self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is None:
            return
        cleaned = normalize_text(data)
        if cleaned:
            self._current_text.append(cleaned)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._current_href is None:
            return
        text = normalize_text(" ".join(self._current_text))
        self.links.append((self._current_href, text))
        self._current_href = None
        self._current_text = []


def normalize_text(text: str) -> str:
    value = unescape(str(text or ""))
    value = value.replace("\ufeff", " ")
    value = value.replace("\u200b", " ")
    value = value.replace("\r", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def compact_blocks(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def to_iso_date(raw: str) -> str:
    raw = normalize_text(raw).replace(",", "")
    for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%d %m %Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def guess_date_from_text(text: str) -> str:
    patterns = [
        r"\b(\d{1,2} [A-Z][a-z]+ \d{4})\b",
        r"\b(\d{1,2} [A-Z][a-z]{2} \d{4})\b",
        r"\b(\d{4}-\d{2}-\d{2})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return to_iso_date(match.group(1))
    return ""


def fetch_url(url: str, timeout: int = 30, retries: int = 3, backoff_seconds: float = 1.0) -> Tuple[bytes, str]:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; CI2026Module4Crawler/1.0)",
                "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                content = response.read()
                content_type = response.headers.get("Content-Type", "")
            return content, content_type
        except HTTPError:
            raise
        except (URLError, OSError, ConnectionResetError) as exc:
            last_error = exc
            if attempt >= retries - 1:
                break
            time.sleep(backoff_seconds * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to fetch URL: {url}")


def fetch_text(url: str, timeout: int = 30) -> str:
    content, _ = fetch_url(url, timeout=timeout)
    return content.decode("utf-8", errors="ignore")


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    scheme = parsed.scheme or "https"
    normalized = parsed._replace(scheme=scheme, path=path, query="", fragment="")
    return normalized.geturl()


def slugify(value: str, default: str = "item") -> str:
    value = normalize_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or default


def build_discovered_source_id(prefix: str, url: str) -> str:
    path = urlparse(url).path.strip("/")
    tail = path.split("/")[-1] if path else "item"
    slug = slugify(tail, default="item")[:48]
    short_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{slug}-{short_hash}"


def parse_robots_sitemaps(seed_url: str) -> List[str]:
    parsed = urlparse(seed_url)
    robots_url = f"{parsed.scheme or 'https'}://{parsed.netloc}/robots.txt"
    try:
        text = fetch_text(robots_url)
    except Exception:
        return []

    sitemap_urls: List[str] = []
    for line in text.splitlines():
        match = re.match(r"(?i)\s*sitemap:\s*(\S+)", line.strip())
        if match:
            sitemap_urls.append(normalize_url(match.group(1)))
    return sitemap_urls


def extract_sitemap_locs(xml_text: str) -> List[str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    urls: List[str] = []
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1].lower()
        if tag == "loc" and elem.text:
            urls.append(normalize_url(elem.text.strip()))
    return urls


def discover_urls_from_sitemaps(seed: DiscoverySeed, cap: int) -> List[str]:
    start_sitemaps: List[str] = [normalize_url(url) for url in seed.sitemap_urls]
    start_sitemaps.extend(url for url in parse_robots_sitemaps(seed.seed_url) if url not in start_sitemaps)
    if not start_sitemaps:
        return []

    found: List[str] = []
    seen_sitemaps: Set[str] = set()
    queue: List[str] = list(start_sitemaps)
    seed_domain = urlparse(seed.seed_url).netloc.lower()

    while queue and len(found) < cap:
        sitemap_url = queue.pop(0)
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)

        try:
            xml_text = fetch_text(sitemap_url)
        except Exception:
            continue

        for loc in extract_sitemap_locs(xml_text):
            parsed = urlparse(loc)
            if parsed.netloc.lower() != seed_domain:
                continue

            path = parsed.path.rstrip("/") or "/"
            if loc.endswith(".xml") or path.endswith(".xml"):
                if loc not in seen_sitemaps:
                    queue.append(loc)
                continue

            allowed = any(path.startswith(prefix) for prefix in seed.allowed_path_prefixes)
            denied = path in seed.denied_path_prefixes
            if allowed and not denied and loc not in found:
                found.append(loc)
                if len(found) >= cap:
                    break

    return found


def infer_extension(url: str, content_type: str) -> str:
    lower_type = (content_type or "").lower()
    if url.lower().endswith(".pdf") or "application/pdf" in lower_type:
        return ".pdf"
    if "html" in lower_type:
        return ".html"
    return ".bin"


def save_raw_file(download_dir: str, source_id: str, content_bytes: bytes, content_type: str, url: str) -> str:
    os.makedirs(download_dir, exist_ok=True)
    extension = infer_extension(url, content_type)
    path = os.path.join(download_dir, f"{source_id}{extension}")
    with open(path, "wb") as f:
        f.write(content_bytes)
    return path


def extract_html(content_bytes: bytes, url: str, title_hint: str = "") -> Tuple[str, str, str]:
    html_text = content_bytes.decode("utf-8", errors="ignore")
    parser = MinimalContentParser()
    parser.feed(html_text)

    title = normalize_text(" ".join(parser.heading_parts)) or normalize_text(" ".join(parser.title_parts)) or title_hint
    body = compact_blocks(" ".join(parser.text_parts))
    published_at = guess_date_from_text(body)

    if not title:
        title = title_hint or urlparse(url).path.strip("/").split("/")[-1] or "untitled"

    return title, body, published_at


def extract_links(content_bytes: bytes, base_url: str) -> List[Tuple[str, str]]:
    html_text = content_bytes.decode("utf-8", errors="ignore")
    parser = LinkParser()
    parser.feed(html_text)

    links: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for href, text in parser.links:
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = normalize_url(urljoin(base_url, href))
        base_parsed = urlparse(base_url)
        abs_parsed = urlparse(absolute)
        if base_parsed.scheme == "https" and abs_parsed.netloc == base_parsed.netloc and abs_parsed.scheme == "http":
            absolute = normalize_url(abs_parsed._replace(scheme="https").geturl())
        if absolute in seen:
            continue
        seen.add(absolute)
        links.append((absolute, text))
    return links


def discover_targets(
    seeds: List[DiscoverySeed],
    known_urls: Set[str],
    max_depth: int = 3,
    per_seed_limit: Optional[int] = None,
) -> List[CrawlTarget]:
    discovered: List[CrawlTarget] = []
    seen_urls = set(known_urls)

    for seed in seeds:
        seed_domain = urlparse(seed.seed_url).netloc.lower()
        cap = per_seed_limit if per_seed_limit is not None else seed.max_links
        matches = 0

        for loc in discover_urls_from_sitemaps(seed, cap=cap):
            if loc in seen_urls:
                continue
            discovered.append(
                CrawlTarget(
                    source_id=build_discovered_source_id(seed.source_id_prefix, loc),
                    url=loc,
                    source_type=seed.source_type,
                    title_hint=seed.title_hint,
                )
            )
            seen_urls.add(loc)
            matches += 1
            if matches >= cap:
                break

        if matches >= cap:
            continue

        queue: List[Tuple[str, int]] = [(normalize_url(seed.seed_url), 0)]
        queued: Set[str] = {normalize_url(seed.seed_url)}

        while queue and matches < cap:
            current_url, depth = queue.pop(0)
            try:
                content_bytes, content_type = fetch_url(current_url)
            except Exception:
                continue

            if "html" not in (content_type or "").lower():
                continue

            for candidate_url, anchor_text in extract_links(content_bytes, current_url):
                parsed = urlparse(candidate_url)
                if parsed.netloc.lower() != seed_domain:
                    continue

                path = parsed.path.rstrip("/") or "/"
                allowed = any(path.startswith(prefix) for prefix in seed.allowed_path_prefixes)
                denied = path in seed.denied_path_prefixes

                if allowed and not denied and candidate_url not in seen_urls:
                    discovered.append(
                        CrawlTarget(
                            source_id=build_discovered_source_id(seed.source_id_prefix, candidate_url),
                            url=candidate_url,
                            source_type=seed.source_type,
                            title_hint=anchor_text or seed.title_hint,
                        )
                    )
                    seen_urls.add(candidate_url)
                    matches += 1
                    if matches >= cap:
                        break

                should_expand = depth < max_depth and candidate_url not in queued
                if should_expand and (allowed or path == "/" or any(path.startswith(prefix.rstrip("/")) for prefix in seed.allowed_path_prefixes)):
                    queue.append((candidate_url, depth + 1))
                    queued.add(candidate_url)

    return discovered


def extract_pdf(content_bytes: bytes, url: str, title_hint: str = "") -> Tuple[str, str, str]:
    title = title_hint or os.path.basename(urlparse(url).path) or "untitled-pdf"
    published_at = guess_date_from_text(title)

    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(content_bytes))
        pages: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            cleaned = compact_blocks(text)
            if cleaned:
                pages.append(cleaned)
        full_text = "\n\n".join(pages)
        if full_text:
            published_at = published_at or guess_date_from_text(full_text)
            title_line = normalize_text(full_text.splitlines()[0] if full_text.splitlines() else "")
            if title_line:
                title = title_line[:220]
            return title, full_text, published_at
    except Exception:
        pass

    fallback = (
        f"PDF source downloaded from {url}. Install pypdf to extract full text automatically. "
        f"Use the title hint and source URL for manual review."
    )
    return title, fallback, published_at


def crawl_target(target: CrawlTarget, download_dir: str = "") -> dict:
    content_bytes, content_type = fetch_url(target.url)
    lower_type = (content_type or "").lower()
    is_pdf = target.url.lower().endswith(".pdf") or "application/pdf" in lower_type
    raw_file_path = ""

    if download_dir:
        raw_file_path = save_raw_file(download_dir, target.source_id, content_bytes, content_type, target.url)

    if is_pdf:
        title, content, published_at = extract_pdf(content_bytes, target.url, target.title_hint)
    else:
        title, content, published_at = extract_html(content_bytes, target.url, target.title_hint)

    return {
        "source_id": target.source_id,
        "title": normalize_text(title),
        "content": compact_blocks(content),
        "source_type": target.source_type,
        "url": target.url,
        "published_at": published_at,
        "raw_file_path": raw_file_path,
    }


def write_csv(rows: Iterable[dict], output_csv: str) -> None:
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def write_json(rows: Iterable[dict], output_json: str) -> None:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(list(rows), f, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl official Singapore fact-check / FAQ sources and export them in vector DB CSV format."
    )
    parser.add_argument(
        "--output-csv",
        default="module4_crawled_vector_docs.csv",
        help="CSV output path with fields: source_id,title,content,source_type,url,published_at",
    )
    parser.add_argument(
        "--output-json",
        default="module4_crawled_vector_docs.json",
        help="Optional JSON mirror of the crawled records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for debugging. 0 means crawl all built-in targets.",
    )
    parser.add_argument(
        "--download-dir",
        default="module4_downloaded_sources",
        help="Optional directory to save raw crawled HTML/PDF files.",
    )
    parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="Disable automatic discovery of additional article and FAQ links from hub pages.",
    )
    parser.add_argument(
        "--discovery-depth",
        type=int,
        default=3,
        help="Recursive discovery depth. Higher means crawling more directory pages to find more links.",
    )
    parser.add_argument(
        "--max-discovered-per-seed",
        type=int,
        default=0,
        help="Optional hard cap of discovered targets per seed. 0 means use the seed's built-in large cap.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = DEFAULT_TARGETS[: args.limit] if args.limit and args.limit > 0 else list(DEFAULT_TARGETS)

    if not args.no_discovery:
        discovered = discover_targets(
            DEFAULT_DISCOVERY_SEEDS,
            known_urls={normalize_url(target.url) for target in targets},
            max_depth=max(0, args.discovery_depth),
            per_seed_limit=(args.max_discovered_per_seed if args.max_discovered_per_seed > 0 else None),
        )
        targets.extend(discovered)
        print(f"discovered_targets: {len(discovered)}")

    rows: List[dict] = []
    failures: List[dict] = []

    for target in targets:
        try:
            row = crawl_target(target, download_dir=args.download_dir)
            if not row["content"]:
                raise ValueError("Empty content after extraction.")
            rows.append(row)
            print(f"[OK] {target.source_id} -> {target.url}")
        except Exception as exc:
            failures.append(
                {
                    "source_id": target.source_id,
                    "url": target.url,
                    "error": str(exc),
                }
            )
            print(f"[FAILED] {target.source_id} -> {exc}")

    write_csv(rows, args.output_csv)
    write_json(rows, args.output_json)

    print(f"\nrecords_written: {len(rows)}")
    print(f"failures: {len(failures)}")
    print(f"output_csv: {args.output_csv}")
    print(f"output_json: {args.output_json}")

    if failures:
        print("\nfailed_targets:")
        print(json.dumps(failures, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
