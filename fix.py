import re
p='src/pages/applications/electrochemical-interfaces.astro'
s=open(p,encoding='utf8',newline='').read()
lines=s.split('\n')
for i,l in enumerate(lines):
    if 'number="3"' in l and 'Math' in l:
        lines[i]='      <Math display number="3" tex={String.raw`\omega_{\mathrm{LO}}^2 - \omega_{\mathrm{TO}}^2 = \frac{4\pi k\,(Z^{*})^2}{\Omega\,\mu\,\varepsilon_\infty}, \qquad \frac{\omega_{\mathrm{LO}}^2}{\omega_{\mathrm{TO}}^2} = \frac{\varepsilon(0)}{\varepsilon_\infty}`} />'
s='\n'.join(lines)
assert not re.search('[\x07\x08\x0b\x0c]', s)
open(p,'w',encoding='utf8',newline='').write(s)
