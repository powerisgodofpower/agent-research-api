"""A small web-research API built with FastAPI.

Keyword search is performed against DuckDuckGo's HTML results page. Pages are
fetched with requests + BeautifulSoup, then summarized extractively (no paid
LLM/API key is required).
"""

from __future__ import annotations

import ipaddress
import os
import re
import secrets
import socket
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Lightweight Web Research API",
    description=(
        "Search the web by keyword or fetch a URL, then return structured "
        "page information and a simple extractive summary."
    ),
    version="1.0.0",
)

USER_AGENT = "LightweightResearchBot/1.0 (+https://render.com)"
REQUEST_TIMEOUT = (4, 10)  # connect timeout, read timeout (seconds)
MAX_PAGE_BYTES = 1_000_000  # do not download unlimited page content
MAX_REDIRECTS = 4
MAX_RESULTS_LIMIT = 5


class ResearchRequest(BaseModel):
    """Input for POST /research. Supply keyword OR url (exactly one)."""

    keyword: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Search term, e.g. 'latest battery recycling technology'.",
    )
    url: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="A public http(s) URL to read directly.",
    )
    max_results: int = Field(default=3, ge=1, le=MAX_RESULTS_LIMIT)


# ---------------------------------------------------------------------------
# URL safety
# ---------------------------------------------------------------------------

def validate_public_url(url: str) -> str:
    """Validate a URL and reject local/private/reserved destinations.

    Redirects are handled manually, and this check is repeated for every hop.
    For a high-risk public production service, also enforce outbound network
    restrictions at the hosting/network layer to mitigate DNS-rebinding races.
    """
    try:
        parsed = urlsplit(url.strip())
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only http:// and https:// URLs are allowed.")
        if not parsed.hostname:
            raise ValueError("The URL must include a hostname.")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("URLs containing credentials are not allowed.")

        # Only standard web ports are needed for this lightweight service.
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        expected_port = 443 if parsed.scheme == "https" else 80
        if port != expected_port:
            raise ValueError("Only standard HTTP (80) and HTTPS (443) ports are allowed.")

        hostname = parsed.hostname.rstrip(".").lower()
        if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
            raise ValueError("Localhost URLs are not allowed.")

        # Resolve all addresses and reject the host if any answer is non-public.
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        if not addresses:
            raise ValueError("The hostname could not be resolved.")
        for address in {entry[4][0] for entry in addresses}:
            ip = ipaddress.ip_address(address.split("%", 1)[0])
            if not ip.is_global:
                raise ValueError("Private, local, or reserved IP addresses are not allowed.")

        # Remove fragments; they are not sent to the server.
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))
    except ValueError:
        raise
    except socket.gaierror as exc:
        raise ValueError("The hostname could not be resolved.") from exc


def fetch_html(url: str) -> tuple[str, str]:
    """Fetch a public HTML page, manually validating every redirect target.

    Returns (final_url, html_text). Raises ValueError for blocked URLs and
    requests.RequestException for network/HTTP failures.
    """
    current_url = url
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
    }

    for _ in range(MAX_REDIRECTS + 1):
        current_url = validate_public_url(current_url)
        response = requests.get(
            current_url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
            stream=True,
        )

        if response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise requests.RequestException("The server returned a redirect without a Location header.")
            current_url = urljoin(current_url, location)
            continue

        try:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                raise requests.RequestException("The URL did not return an HTML page.")

            content_length = response.headers.get("Content-Length")
            if content_length and content_length.isdigit() and int(content_length) > MAX_PAGE_BYTES:
                raise requests.RequestException("The page is larger than the allowed download size.")

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=16_384):
                if not chunk:
                    continue
                remaining = MAX_PAGE_BYTES - total
                chunks.append(chunk[:remaining])
                total += min(len(chunk), remaining)
                if total >= MAX_PAGE_BYTES:
                    break

            raw = b"".join(chunks)
            encoding = response.encoding or "utf-8"
            return current_url, raw.decode(encoding, errors="replace")
        finally:
            response.close()

    raise requests.RequestException("The page exceeded the maximum number of redirects.")


# ---------------------------------------------------------------------------
# Search and extraction
# ---------------------------------------------------------------------------

def search_duckduckgo(keyword: str, max_results: int) -> list[dict[str, str]]:
    """Get a small list of search results from DuckDuckGo's HTML interface."""
    search_url = "https://html.duckduckgo.com/html/?q=" + quote_plus(keyword)
    response = requests.get(
        search_url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    for anchor in soup.select("a.result__a"):
        href = anchor.get("href", "").strip()
        if not href:
            continue

        # DuckDuckGo may wrap the destination in an /l/?uddg=... URL.
        if "duckduckgo.com/l/" in href:
            wrapped = parse_qs(urlsplit(href).query).get("uddg", [None])[0]
            if wrapped:
                href = unquote(wrapped)

        href = urljoin("https://duckduckgo.com", href)
        title = anchor.get_text(" ", strip=True)
        result_box = anchor.find_parent(class_="result")
        snippet_node = result_box.select_one(".result__snippet") if result_box else None
        snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""

        if href and href not in seen:
            seen.add(href)
            results.append({"title": title[:300], "url": href, "snippet": snippet[:500]})
        if len(results) >= max_results * 2:
            break

    return results


def clean_page(html: str) -> tuple[str, str, str]:
    """Extract a page title, readable paragraphs, and meta description."""
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(
        ["script", "style", "noscript", "svg", "canvas", "iframe", "nav", "footer", "header", "form"]
    ):
        tag.decompose()

    title = ""
    if soup.title:
        title = soup.title.get_text(" ", strip=True)
    if not title:
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title:
            title = str(og_title.get("content", "")).strip()

    description = ""
    meta_description = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    if meta_description:
        description = str(meta_description.get("content", "")).strip()

    paragraphs: list[str] = []
    seen: set[str] = set()
    for node in soup.find_all(["p", "li"]):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if len(text) >= 45 and text not in seen:
            seen.add(text)
            paragraphs.append(text)

    body_text = "\n".join(paragraphs)
    if not body_text:
        body_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

    return title[:300], body_text[:12_000], description[:800]


def make_extractive_summary(text: str, description: str = "") -> str:
    """Create a short extractive summary without a generative AI service."""
    source = text.strip() or description.strip()
    if not source:
        return "本文から要約できるテキストを抽出できませんでした。"

    # Sentence splitting supports both English and Japanese punctuation.
    sentences = re.split(r"(?<=[.!?。！？])\s*|\n+", source)
    selected: list[str] = []
    for sentence in sentences:
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if 45 <= len(sentence) <= 500 and sentence not in selected:
            selected.append(sentence)
        if len(selected) == 3:
            break

    if selected:
        return " ".join(selected)[:900]
    return source[:500]


def build_result(url: str, fallback_title: str = "", fallback_snippet: str = "") -> dict[str, Any]:
    """Fetch and parse one result. Keep per-page failures in the JSON output."""
    try:
        final_url, html = fetch_html(url)
        title, text, description = clean_page(html)
        summary = make_extractive_summary(text, description or fallback_snippet)
        excerpt = text[:2500] if text else (description or fallback_snippet)[:2500]
        return {
            "title": title or fallback_title or final_url,
            "url": final_url,
            "summary": summary,
            "content_excerpt": excerpt,
            "status": "ok",
        }
    except ValueError as exc:
        return {
            "title": fallback_title or url,
            "url": url,
            "summary": fallback_snippet or "取得をスキップしました。",
            "content_excerpt": "",
            "status": "blocked_or_invalid_url",
            "error": str(exc),
        }
    except requests.RequestException as exc:
        return {
            "title": fallback_title or url,
            "url": url,
            "summary": fallback_snippet or "ページを取得できませんでした。",
            "content_excerpt": "",
            "status": "fetch_error",
            "error": str(exc)[:300],
        }


# ---------------------------------------------------------------------------
# Optional API-key protection
# ---------------------------------------------------------------------------

async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """If API_KEY is set in the environment, require X-API-Key on requests."""
    expected = os.getenv("API_KEY")
    if expected and (not x_api_key or not secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "Lightweight Web Research API", "health": "ok", "docs": "/docs"}


@app.post("/research", dependencies=[Depends(verify_api_key)])
async def research(request: ResearchRequest) -> dict[str, Any]:
    """Research one keyword or one URL and return clean JSON."""
    keyword = request.keyword.strip() if request.keyword else ""
    target_url = request.url.strip() if request.url else ""

    if bool(keyword) == bool(target_url):
        raise HTTPException(
            status_code=422,
            detail="Provide exactly one of 'keyword' or 'url'.",
        )

    results: list[dict[str, Any]] = []

    if target_url:
        # Validate before starting the request so invalid input gets a clear 400.
        try:
            validate_public_url(target_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        results.append(build_result(target_url))
        query_info = {"type": "url", "url": target_url}

    else:
        if len(keyword) < 2:
            raise HTTPException(status_code=422, detail="'keyword' must contain at least 2 characters.")
        try:
            search_results = search_duckduckgo(keyword, request.max_results)
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"The search provider could not be reached: {str(exc)[:200]}",
            ) from exc

        for item in search_results:
            results.append(
                build_result(
                    item["url"],
                    fallback_title=item["title"],
                    fallback_snippet=item["snippet"],
                )
            )
            if len(results) >= request.max_results:
                break
        query_info = {"type": "keyword", "keyword": keyword}

    return {
        "query": query_info,
        "count": len(results),
        "results": results,
        "summary_method": "extractive (no LLM/API key)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
