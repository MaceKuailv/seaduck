# Contributing to documentaion
We are using [Jupyter Book](https://jupyterbook.org/en/stable/intro.html#)📙 to build the documentation. The online documentaion is hosted by [github pages](https://pages.github.com/). 
## Add/Modify
[Jupyter Book](https://jupyterbook.org/en/stable/intro.html#)📗 support several different kinds of files as input, including [.md files](https://jupyterbook.org/en/stable/reference/cheatsheet.html#tags)⬇️ [.rst files](https://docutils.sourceforge.io/docs/user/rst/cheatsheet.html)📜, [.ipynb files](https://www.ibm.com/docs/en/watson-studio-local/1.2.3?topic=notebooks-markdown-jupyter-cheatsheet)🐍, etc. Executable ones generally requires a slightly different procedures to the narrative files. Here are some instruction on how to work with each of them. 



## Text files
Adding  files (markdown, reStructuredText, etc) are not very different from contributing code. 

1. [Clone the git repository to your local machine](./use_git)
2. [Build the environment](prep_env.md). If you do not run this step, the API references and notebooks can not be properly build. 
3. Add or change the files. A useful cheat sheet can be found here: markdown, reStructuredText
4. If you added a new file,  in `docs/_toc.yml`, add the name of your new file in the corresponding location. 
5. Change directory to seaduck, and run
```
make docs-build
```
6. Go to `seaduck/docs/_build/html`, you can either open index.html or the html file with the same name to the one that you have made changes to. See if it looks right to you. If not, iterate a little. Send me a message if you need help. 
7. When you are happy with the result, you can make a pull request. After approval, your changes will be {ref}`deployed <deploy_doc>`. 
## Notebooks that could be run any where
## Sciserver notebooks
(deploy_doc)=
## Deploy documentation
