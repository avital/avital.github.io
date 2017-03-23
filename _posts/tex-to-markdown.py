import re
import sys

def main(content, pdf_filename):
    flags = re.DOTALL
    content = re.sub(r'\$', '$$', content, 0, flags)

    def replace_all(pattern, to, content):
        while True:
            content, times = re.subn(pattern, to, content, 0, flags)
            if times == 0:
                return content

    content = replace_all(r'(\\begin\{enumerate\}.*)\\item(.*?\\end\{enumerate\})', '\g<1>1. \\2', content)
    content = replace_all(r'(\\begin\{itemize\}.*)\\item(.*?\\end\{itemize\})', '\g<1>* \\2', content)

    content = re.sub(r'\\begin{enumerate}', '', content, 0, flags)
    content = re.sub(r'\\end{enumerate}', '', content, 0, flags)
    content = re.sub(r'\\begin{itemize}', '', content, 0, flags)
    content = re.sub(r'\\end{itemize}', '', content, 0, flags)
    content = re.sub(r'\\emph{(.*?)}', '**\\1**', content, 0, flags)
    content = re.sub(r'\\end\{document\}', '', content, 0, flags)
    content = re.sub(r'.*\\maketitle', '', content, 0, flags)
    content = re.sub(r'\\section', '<!--more-->\n(This note is also available as a [PDF](/assets/{0}).)\n\\section'.format(pdf_filename), content, 1, flags)
    content = re.sub(r'\\section{(.*?)}', '## \\1', content, 0, flags)

    # Weird but we need 6 backslashes for a newline
    content = replace_all(r'(\\begin\{align\}.*)[^\\]\\\\[^\\](.*?\\end\{align\})', '\\1 \\\\\\\\\\\\ \\2', content)

    return content

if len(sys.argv) > 1 and sys.argv[1] == 'test':
    ex = 'Such as $x^2$ and $\sqrt(x)$, right'
    print(main(ex))

    ex = """
\emph{Before}
\\begin{enumerate}
\\item One
\\item Two
\\end{enumerate}
\emph{Between}
\\begin{itemize}
\\item One
\\item Two
\\end{itemize}
After
    """
    print(main(ex))

    ex = """
\\begin{align}
Foo & = & Bar
\\end{align}
    """
    print(main(ex))

elif __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: python tex-to-markdown.py <PDF filename>")
        exit(1)
    print(main(sys.stdin.read(), sys.argv[1]))

