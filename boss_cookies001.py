import random
from subprocess import Popen
from urllib import parse

import execjs
import requests

Popen.__init__.__kwdefaults__['encoding'] = "utf-8"


class BossCookieSpider:

    def __init__(self):
        self.core = """
    window = {
    document: {
        cookie: "",
        createElement: function(tag) {
            if (tag == "canvas") {
                return canvas
            } else if (tag == "caption") {
                return {
                    tagName: "CAPTION"
                }
            }
            
        },
        getElementById: function() {
            return false
        },
        title: ""
    },
    moveBy: function () {},
    moveTo: function () {},
    open: function(){},
    dispatchEvent: function(){return true},
    screen: {
        availHeight: 824,
        availWidth: 1536
    },
    navigator: {
        cookieEnabled: true,
        language: "zh-CN",
        userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36",
        appVersion: "5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36"
    },
    decodeURI: global.decodeURI,
    location: {
        hostname: "www.zhipin.com",
        href: "https://www.zhipin.com/"
    },
    OfflineAudioContext: function () {
        this.createOscillator = function() {
            return {
                frequency: {
                    setValueAtTime: function() {}
                },
                connect: function (){},
                start: function (){},
            }
        },
        this.createDynamicsCompressor = function() {
            return {
                threshold: {
                    setValueAtTime: function () {},
                },
                setValueAtTime: function () {},
                knee: {
                    setValueAtTime: function () {},
                },
                ratio: {
                    setValueAtTime: function () {},
                },
                reduction: {
                    setValueAtTime: function () {},
                },
                attack: {
                    setValueAtTime: function () {},
                },
                release: {
                    setValueAtTime: function () {},
                },
                connect: function (){},
            }
        },
        this.startRendering = function (){}
    },
    eval: global.eval,
    history: {length: 1},
    outerHeight: 824,
    innerHeight: 150,
    outerWidth: 1536,
    innerWidth: 0,
    Math: global.Math,
    Date: global.Date,
}
window.open.toString = function (){return "function open() { [native code] }"}
document = window.document
navigator = window.navigator
screen = window.screen
canvas = {
    getContext: function getContext() {
        return CanvasRenderingContext2D
    },
    toDataURL: function toDataURL() {
        // 实际为canvas画布填充了“"k54kk5cA4*"”字样后，转成的图片链接
        return "data:image/png;base64,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"
    },
}
CanvasRenderingContext2D = {
    fillRect: function () {},
    fillText: function () {}
}
localStorage = {
    removeItem: function (key) {
        delete this[key]
    },
    getItem: function (key) {
        return this[key] ? this[key]: null;
    },
    setItem: function (key, value) {
        this[key] = "" + value;
    },
};
sessionStorage = {}
setInterval = window.setInterval = function (){}
setInterval.toString = function(){return "function setInterval() { [native code] }"}
setTimeout = function (){}
top = window.top = window
global = undefined;
child_process = undefined;
closed = {
    __proto__: ( 1>>3 >4 )["__proto__"]
}
function get_cookie(seed, ts, code) {
    var Buffer;
    process = undefined;
    function CustomEvent() {}
    eval(code);
    cookie = encodeURIComponent(new ABC().z(seed, parseInt(ts)+(480+new Date().getTimezoneOffset())*60*1000))
    console.log({cookie, cookie})
    return {cookie, cookie};
}
function encryption(seed, ts) {
    var code = new ABC().z(seed, parseInt(ts) + (480 + new Date().getTimezoneOffset()) * 60 * 1000);
    return encodeURIComponent(code)
}
        """
        self.core = """
        var base64EncodeChars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/',
    base64DecodeChars = new Array((-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), (-1), 62, (-1), (-1), (-1), 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, (-1), (-1), (-1), (-1), (-1), (-1), (-1), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, (-1), (-1), (-1), (-1), (-1), (-1), 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, (-1), (-1), (-1), (-1), (-1));
base64encode = function (e) {
    var r, a, c, h, o, t;
    for (c = e.length, a = 0, r = ''; a < c;) {
        if (h = 255 & e.charCodeAt(a++), a == c) {
            r += base64EncodeChars.charAt(h >> 2),
                r += base64EncodeChars.charAt((3 & h) << 4),
                r += '==';
            break
        }
        if (o = e.charCodeAt(a++), a == c) {
            r += base64EncodeChars.charAt(h >> 2),
                r += base64EncodeChars.charAt((3 & h) << 4 | (240 & o) >> 4),
                r += base64EncodeChars.charAt((15 & o) << 2),
                r += '=';
            break
        }
        t = e.charCodeAt(a++),
            r += base64EncodeChars.charAt(h >> 2),
            r += base64EncodeChars.charAt((3 & h) << 4 | (240 & o) >> 4),
            r += base64EncodeChars.charAt((15 & o) << 2 | (192 & t) >> 6),
            r += base64EncodeChars.charAt(63 & t)
    }
    return r
}
base64decode = function (e) {
    var r, a, c, h, o, t, d;
    for (t = e.length, o = 0, d = ''; o < t;) {
        do r = base64DecodeChars[255 & e.charCodeAt(o++)];
        while (o < t && r == -1);
        if (r == -1) break;
        do a = base64DecodeChars[255 & e.charCodeAt(o++)];
        while (o < t && a == -1);
        if (a == -1) break;
        d += String.fromCharCode(r << 2 | (48 & a) >> 4);
        do {
            if (c = 255 & e.charCodeAt(o++), 61 == c) return d;
            c = base64DecodeChars[c]
        } while (o < t && c == -1);
        if (c == -1) break;
        d += String.fromCharCode((15 & a) << 4 | (60 & c) >> 2);
        do {
            if (h = 255 & e.charCodeAt(o++), 61 == h) return d;
            h = base64DecodeChars[h]
        } while (o < t && h == -1);
        if (h == -1) break;
        d += String.fromCharCode((3 & c) << 6 | h)
    }
    return d
}
window = {
    document: {
        cookie: "",
        createElement: function (tag) {
            if (tag == "canvas") {
                return canvas
            } else if (tag == "caption") {
                return {
                    tagName: "CAPTION"
                }
            }

        },
        getElementById: function () {
            return false
        },
        title: ""
    },
    moveBy: function () {
    },
    moveTo: function () {
    },
    open: function () {
    },
    dispatchEvent: function () {
        return true
    },
    screen: {
        availHeight: 824,
        availWidth: 1536
    },
    navigator: {
        cookieEnabled: true,
        language: "zh-CN",
        userAgent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36",
        appVersion: "5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36"
    },
    decodeURI: global.decodeURI,
    location: {
        hostname: "www.zhipin.com",
        href: "https://www.zhipin.com/"
    },
    OfflineAudioContext: function () {
        this.createOscillator = function () {
            return {
                frequency: {
                    setValueAtTime: function () {
                    }
                },
                connect: function () {
                },
                start: function () {
                },
            }
        },
            this.createDynamicsCompressor = function () {
                return {
                    threshold: {
                        setValueAtTime: function () {
                        },
                    },
                    setValueAtTime: function () {
                    },
                    knee: {
                        setValueAtTime: function () {
                        },
                    },
                    ratio: {
                        setValueAtTime: function () {
                        },
                    },
                    reduction: {
                        setValueAtTime: function () {
                        },
                    },
                    attack: {
                        setValueAtTime: function () {
                        },
                    },
                    release: {
                        setValueAtTime: function () {
                        },
                    },
                    connect: function () {
                    },
                }
            },
            this.startRendering = function () {
            }
    },
    eval: global.eval,
    history: {length: 1},
    outerHeight: 824,
    innerHeight: 150,
    outerWidth: 1536,
    innerWidth: 0,
    Math: global.Math,
    Date: global.Date,
    Function:Function
}
window.open.toString = function () {
    return "function open() { [native code] }"
}
atob = base64decode
document = window.document
navigator = window.navigator
screen = window.screen
canvas = {
    getContext: function getContext() {
        return CanvasRenderingContext2D
    },
    toDataURL: function toDataURL() {
        // 实际为canvas画布填充了“"k54kk5cA4*"”字样后，转成的图片链接
        return "data:image/png;base64,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"
    },
}
CanvasRenderingContext2D = {
    fillRect: function () {
    },
    fillText: function () {
    }
}
localStorage = {
    removeItem: function (key) {
        delete this[key]
    },
    getItem: function (key) {
        return this[key] ? this[key] : null;
    },
    setItem: function (key, value) {
        this[key] = "" + value;
    },
};
sessionStorage = {}
setInterval = window.setInterval = function () {
}
setInterval.toString = function () {
    return "function setInterval() { [native code] }"
}
setTimeout = function () {
}
top = window.top = window
global = undefined;
child_process = undefined;
closed = {
    __proto__: (1 >> 3 > 4)["__proto__"]
}

function get_cookie(seed, ts, code) {
    var Buffer;
    process = undefined;

    function CustomEvent() {
    }

    eval(code);
    cookie = encodeURIComponent(new ABC().z(seed, parseInt(ts) + (480 + new Date().getTimezoneOffset()) * 60 * 1000))
    console.log({cookie, cookie})
    return {cookie, cookie};
}

function encryption(seed, ts) {
    var code = new ABC().z(seed, parseInt(ts) + (480 + new Date().getTimezoneOffset()) * 60 * 1000);
    return encodeURIComponent(code)
}
        """

    def main(self):
        headers = {
            "user-agent": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/{random.randint(1, 999)}.36 (KHTML, like Gecko) Chrome/81.0.4044.129 Safari/537.36",
        }
        url = f"https://www.zhipin.com/job_detail/"
        response = requests.get(url, headers=headers)
        self.cookies_generate(response)

    def cookies_generate(self, response):
        query_str = parse.urlparse(response.url).query
        query_dict = {i.split("=")[0]: i.split("=")[1] for i in query_str.split("&")}
        js_name = query_dict.get("name")
        js_url = f"https://www.zhipin.com/web/common/security-js/{js_name}.js"
        js_res = requests.get(js_url)
        js_text = js_res.text
        js_text = js_text.replace("typeof process", "typeof child_process")
        seed = parse.unquote(query_dict.get("seed"))
        ts = query_dict.get("ts")
        new_js_text = self.core + js_text
        new_js = execjs.compile(new_js_text)
        code = new_js.call("encryption", seed, ts)
        headers = {
            "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
            "cookie": f"__zp_stoken__={code};"
        }
        url = "https://www.zhipin.com/c101280600-p100120/"
        response = requests.get(url, headers=headers)
        print(response.text)
        print("js 名字", js_name)
        print("cookie",code)
        "d2333fb0"
        pass


def adc(yuming):
    res = requests.get(yuming)
    json = res.json()
    return json


if __name__ == '__main__':
    boss_zhipin_cookie = BossCookieSpider()
    boss_zhipin_cookie.main()
