"""Unit tests for canonicalisation (run: python3 test_urlfeat.py)."""
from urlfeat import canonical, host_view, reg_domain, presentation_feats, split
cases = {
    "https://www.Example.com:443/a/b/": "example.com/a/b",
    "HTTP://www.example.com/": "example.com",
    "example.com": "example.com",
    "x.com/a?id=1&reg=2&not=3&copy=5": "x.com/a?id=1&reg=2&not=3&copy=5",
    "x.com/a?id=1&amp;b=2": "x.com/a?id=1&b=2",
    "http://user@www.paypal.com.evil.tk/x": "user@paypal.com.evil.tk/x",
    "x.com/?q=/": "x.com/?q=/",
    "x.com/a//": "x.com/a/",
    "http://x.com:8080/a": "x.com:8080/a",
    "http://[2001:db8::1]:443/a": "[2001:db8::1]/a",
    "http://x.com/a#frag/": "x.com/a#frag/",
}
for u, want in cases.items():
    got = canonical(u)
    assert got == want, (u, got, want)
assert host_view("http://user@www.Ex.com:8080/a?b") == "ex.com"
assert reg_domain("http://a.b.blogspot.com/x") == "b.blogspot.com"
assert reg_domain("http://a.b.blogspot.com/x", private=False) == "blogspot.com"
assert presentation_feats("https://www.A.com/") == [1, 1, 1, 0, 1]
print("all", len(cases) + 4, "tests passed")
