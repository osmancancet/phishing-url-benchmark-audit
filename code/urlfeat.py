"""URL parsing, canonicalisation and hand-crafted feature groups."""
import html, re
from urllib.parse import urlsplit, unquote
import numpy as np
import pandas as pd
import tldextract

_EXT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)  # bundled PSL snapshot, no network
_EXTP = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_IP = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _authority_split(rest):
    """Split 'authority + remainder' at the first '/', '?' or '#'."""
    m = re.search(r"[/?#]", rest)
    return (rest, "") if m is None else (rest[:m.start()], rest[m.start():])


def split(url):
    """Return (scheme, host, port, path, query, fragment, userinfo) without assuming a scheme is present.
    Hand-written so that userinfo, IPv6 brackets and query text are preserved exactly."""
    s = url.strip()
    m = _SCHEME.match(s)
    scheme = m.group(0)[:-3].lower() if m else ""
    rest = s[m.end():] if m else s
    auth, rem = _authority_split(rest)
    userinfo = ""
    if "@" in auth:
        userinfo, auth = auth.rsplit("@", 1)
    port = None
    if auth.startswith("["):
        j = auth.find("]")
        host, tail = auth[:j + 1], auth[j + 1:]
        if tail.startswith(":") and tail[1:].isdigit():
            port = int(tail[1:])
    else:
        host = auth
        if ":" in auth:
            h, p = auth.rsplit(":", 1)
            if p.isdigit():
                host, port = h, int(p)
    frag = ""
    if "#" in rem:
        rem, frag = rem.split("#", 1)
    query = ""
    if "?" in rem:
        rem, query = rem.split("?", 1)
    return scheme, host, port, rem, query, frag, userinfo


def canonical(url):
    """Remove presentation artefacts only: the scheme, one leading 'www.' label, host case, a trailing dot on the
    host, the default ports 80/443, HTML-escaped ampersands ('&amp;' -> '&'), and a single trailing '/' at the end
    of the path when no query or fragment follows. Userinfo, other ports, path, query and fragment are kept verbatim."""
    u = url.strip().replace("&amp;", "&")
    scheme, host, port, path, query, frag, userinfo = split(u)
    host = host.lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if port in (80, 443):
        port = None
    out = (userinfo + "@" if userinfo else "") + host + (f":{port}" if port is not None else "")
    if not query and not frag and path.endswith("/"):
        path = path[:-1]
    out += path
    if query:
        out += "?" + query
    if frag:
        out += "#" + frag
    return out


def host_view(url):
    """Host-only view: canonical host (no userinfo, port, path, query or fragment). Removes path-sampling artefacts
    (homepage-only legitimate sets) as well as presentation artefacts."""
    _, host, _, _, _, _, _ = split(canonical(url))
    return host


def reg_domain(url, private=True):
    """Registered domain (eTLD+1). With private=True, PSL private suffixes (blogspot.com, 000webhostapp.com,
    github.io, ...) count as public suffixes, so unrelated customer sites are not merged into one group."""
    _, host, _, _, _, _, _ = split(url)
    host = host.lower().rstrip(".")
    if _IP.match(host) or host.startswith("["):
        return host
    e = (_EXTP if private else _EXT)(host)
    return (e.domain + "." + e.suffix) if e.suffix else (e.domain or host)


PRESENTATION = ["has_scheme", "has_www", "trailing_slash", "html_entity", "upper_in_host"]
SCHEME_TYPE = ["is_https"]
STRUCTURE = ["has_path", "path_depth", "has_query", "n_params", "url_len"]


def presentation_feats(url):
    """Recording conventions only (see PRESENTATION). The https flag is kept apart (scheme_feats)."""
    s = url.strip()
    scheme, host, port, path, query, frag, userinfo = split(s)
    return [int(scheme != ""), int(host.lower().startswith("www.")), int(s.endswith("/")), int("&amp;" in s),
            int(any(c.isupper() for c in host))]


def scheme_feats(url):
    scheme = split(url)[0]
    return [int(scheme == "https")]


def structure_feats(url):
    c = canonical(url)
    scheme, host, port, path, query, frag, userinfo = split(c)
    segs = [x for x in path.split("/") if x]
    return [int(len(segs) > 0), len(segs), int(query != ""), (query.count("&") + 1) if query else 0, len(c)]


SUSP = ["login", "signin", "verify", "account", "update", "secure", "bank", "paypal", "webscr", "confirm", "password", "ebay", "apple", "wallet"]
LEXICAL = ["has_port", "has_userinfo", "host_len", "n_dots_host", "n_hyphen_host", "n_digits_host", "is_ip", "n_subdomains", "tld_len",
           "path_len", "n_digits_path", "n_special", "has_at", "has_dslash_path", "pct_encoded", "entropy",
           "n_susp_words", "longest_token", "ratio_digits"]


def _entropy(s):
    if not s:
        return 0.0
    _, c = np.unique(list(s), return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def lexical_feats(url):
    """Content features computed on the canonical form, so they are blind to presentation artefacts."""
    c = canonical(url)
    scheme, host, port, path, query, frag, userinfo = split(c)
    e = _EXT(host) if not (_IP.match(host) or host.startswith("[")) else None
    sub = e.subdomain if e else ""
    toks = re.split(r"[^a-zA-Z0-9]", c)
    low = c.lower()
    return [
        int(port is not None), int(userinfo != ""),
        len(host), host.count("."), host.count("-"), sum(ch.isdigit() for ch in host), int(bool(_IP.match(host))),
        (sub.count(".") + 1) if sub else 0, len(e.suffix) if e else 0,
        len(path), sum(ch.isdigit() for ch in path), sum(not ch.isalnum() for ch in c), int("@" in c),
        int("//" in path), c.count("%"), _entropy(c), sum(w in low for w in SUSP),
        max((len(t) for t in toks), default=0), sum(ch.isdigit() for ch in c) / max(1, len(c)),
    ]


def matrix(urls, fn):
    return np.asarray([fn(u) for u in urls], dtype=float)
