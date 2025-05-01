import tldextract
import re
from urllib.parse import urlparse

def extract_features(url):
    ext = tldextract.extract(url)
    parsed = urlparse(url)

    subdomain = ext.subdomain
    domain = ext.domain
    suffix = ext.suffix

    path = parsed.path
    query = parsed.query

    url_parts = url.split(".")
    num_segments = len(url_parts)  # Number of segments in the URL
    num_subdomains = len(subdomain.split(".")) if subdomain else 0

    features = {
        "url_length": len(url),
        "hostname_length": len(parsed.netloc),
        "path_length": len(path),
        "query_length": len(query),
        "domain_length": len(domain),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special": len(re.findall(r"[^\w\s]", url)),
        "has_ip": bool(re.match(r"(http[s]?://)?\d{1,3}(\.\d{1,3}){3}", url)),
        "subdomain_length": len(subdomain),
        "is_https": parsed.scheme == "https",
        "num_dots": url.count("."),
        "num_params": query.count("&") + 1 if query else 0,
        "has_at_symbol": "@" in url,
        "has_redirect": "//" in url[len(parsed.scheme)+3:],  # check for '//' after 'http[s]://'
        "has_long_subdomain": len(subdomain) > 20,
        "has_suspicious_word": any(word in url.lower() for word in ["login", "verify", "account", "secure", "update", "bank",
    "signin", "paypal", "ebay", "password", "confirm", "security",
    "invoice", "support", "billing", "webscr", "redirect", "token"]),
        "domain_in_path": domain.lower() in path.lower(),
        "is_short_url": bool(re.search(r"(bit\.ly|goo\.gl|tinyurl\.com|t\.co|ow\.ly)", url)),
        "num_segments": num_segments,
        "num_subdomains": num_subdomains,
        "sus_suffix" : any(word in suffix.lower() for word in {"com", "org", "net", "edu", "gov", "co", "io", "us"})
    }

    return features