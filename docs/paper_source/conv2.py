import re, html
base="/private/tmp/claude-501/-Users-stevenyang/3952dc21-32c7-4dd5-adb6-1c429cfe1c9c/scratchpad/paper/"
s=open(base+"v4.html",encoding="utf-8").read()
s=re.sub(r"(?is)<script.*?</script>","",s)
s=re.sub(r"(?is)<style.*?</style>","",s)
store=[]
def mrep(m):
    a=re.search(r'alttext="(.*?)"',m.group(0),re.S)
    store.append(html.unescape(a.group(1)) if a else "")
    return "\x00%d\x00"%(len(store)-1)
s=re.sub(r"(?is)<math\b.*?</math>", mrep, s)
for t in ["p","div","section","h1","h2","h3","h4","h5","h6","li","tr","table","figure","figcaption","br","blockquote"]:
    s=re.sub(r"(?is)</?%s\b[^>]*>"%t, "\n", s)
s=re.sub(r"(?is)</?t[dh]\b[^>]*>", " | ", s)
s=re.sub(r"(?is)<[^>]+>","",s)
s=html.unescape(s)
s=re.sub(r"\x00(\d+)\x00", lambda m: " $"+store[int(m.group(1))]+"$ ", s)
s=re.sub(r"[ \t\xa0]+"," ",s)
s=re.sub(r"\n\s*\n\s*\n+","\n\n",s)
open(base+"v4.txt","w",encoding="utf-8").write(s)
print(len(s.split()))
