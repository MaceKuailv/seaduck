# Contributing to documentation🦆

We are using [Jupyter Book](https://jupyterbook.org/en/stable/intro.html#)📙 to build the documentation. The online documentaion is hosted by [github pages](https://pages.github.com/).

## Add/Modify

[Jupyter Book](https://jupyterbook.org/en/stable/intro.html#)📗 support several different kinds of files as input, including [.md files](https://jupyterbook.org/en/stable/reference/cheatsheet.html#tags)⬇️ [.rst files](https://docutils.sourceforge.io/docs/user/rst/cheatsheet.html)📜, [.ipynb files](https://www.ibm.com/docs/en/watson-studio-local/1.2.3?topic=notebooks-markdown-jupyter-cheatsheet)🐍, etc. Executable ones generally requires a slightly different procedures to the narrative files. Here are some instruction on how to work with each of them.

(text_file)=

## Text files

Adding  files (markdown, reStructuredText, etc) are not very different from contributing code.

1. [Clone the git repository to your local machine](./use_git)
1. [Build the environment](prep_env.md). If you do not run this step, the API references and notebooks can not be properly build.
1. Add or change the files. A useful cheat sheet can be found here: [markdown/MyST](https://jupyterbook.org/en/stable/reference/cheatsheet.html#tags), [reStructuredText](https://docutils.sourceforge.io/docs/user/rst/cheatsheet.html). [Commit](use_git.md) the changes you did along the way.
1. If you added a new file,  in `docs/_toc.yml`, add the name of your new file in the corresponding location. This [tutorial](https://jupyterbook.org/en/stable/structure/toc.html) will be helpful, if the file structure is not self-explanatory.
1. Change directory to seaduck, and run

```none
make docs-build
```

6. Go to `seaduck/docs/_build/html`, you can either open `index.html` or the html file with the same name to the one that you have made changes to. See if it looks right to you. If not, iterate a little. Send me a message if you need help.
1. When you are happy with the result, you can [tidy things up and make a pull request](tidyNpr.md). After approval, your changes will be {ref}`deployed <deploy_doc>`.

## Notebooks that could be run any where

As a package that works with oceanographic datasets, almost all the demonstrations requires some supporting data. `seaduck` provides several datasets available to be downloaded everywhere. It can be accessed simply by

```python
ds = seaduck.utils.get_dataset(name)
```

Another option is to generate the dataset using mathematical expressions out of thin air. Since this kind of notebook is executed whenever the github action is triggered, it is preferrable that these notebooks run very fast. For example, do not perform heavy calculations in those notebooks and please do not install packages within them.

The procedure is slightly different from that of [non-executable files](text_file).

1. Follow step 1 to 6 in the [previous](text_file) section.
1. Run

```none
make qa
```

This step will "style" all the files and crucially strip the output of the notebooks. The first time running it is almost certainly going to fail. Don't worry, `pre-commit`, the thing under the hood, will automatically fix most of the problems.

Now, run it again. If this time it still has an error, look at the error message and see what you can do.
3\. Now, you can [tidy things up and make a pull request](tidyNpr.md)

## Cooler (Sciserver) notebooks

Cool stuff are not always portable. The ocean 🌊 is an example of that.
Say you have something really cool you want to demonstrate, but the dataset it is based on is to large to distribute or it simply takes too long to run. Wouldn't it be nice if we could have a cloud platform that host a bunch of ocean dataset that is free for everyone to use? It would be even better if the packages I need as an oceanographer is readily installed and I don't have to worry about a thing.
You can use [Sciserver](https://sciserver.org/)! (Am I too dramatic?). Sciserver is also the home base of [oceanspy](https://oceanspy.readthedocs.io/en/latest/), a package that will make your life so much easier as a oceanographer. After registering on sciserver (you can follow this youtube tutorial here(Tom, if you are reading this, can you send me the link?)), you can simply call this oceanspy function

```python
import ocenspy as ospy
od = ospy.open_oceandataset.from_catalog('NameOfDataset')
ds = od._ds
```

Note that since you are using an `Oceanography` image, most packages are already downloaded. We still need two dependencies just to convert the notebooks.

```
pip install -U jupyter-book
pip install jupytext
```

And that's it, you don't have to follow the steps of preparing environment.

Now, follow these steps:

1. [Fork and clone](use_git.md) this repo adjacent to the seaduck directory

```shell
git clone https://github.com/YourGithubNickname/seaduck_sciserver_notebook.git
```

By "adjacent", I mean the file structure looks like:

```none
parent_dir
- seaduck
- seaduck_sciserver_notebook
```

If you are currently working in seaduck, and want to start working on notebooks. You can run

```shell
cd ..
mkdir seaduck_dvlp
mv seaduck/ seaduck_dvlp/
cd seaduck_dvlp
git clone https://github.com/YourGithubNickname/seaduck_sciserver_notebook.git
```

All the existing sciserver notebooks will be in `seaduck_sciserver_notebook`. If you want to create new ones, put them in there as well.
2\. Have all the fun with your notebooks. However, whenever you plot, **always** use `plt.show()`.
3\. Install `Jupytext` for converting notebooks into markdown.

```shell
pip install jupytext
```

4\. In `seaduck_sciserver_notebook`, run the python script

```shell
python convert_ipynb.py
```

> This step will add when and which version the notebook was last run on. It will search the file with the string **Wenrui Jiang** (How egoistic?!), and put the information in the next line. It is a bit ad hoc. If I am not the author of the notebook, simply include: "Wenrui Jiang is a good boy" or something like that after putting your name.

5. You will realize that the markdown files created may not be able to render properly. This is because the new plots you have are local, but the link we put in the markdown files are what they would look like if the plots are already uploaded. Now, commit all the changes and make a pull request to the `seaduck_sciserver_notebook` repo. Once the changes are merged into the `main` branch. Open the file and see if it look as you intended. If so,

```shell
cp *.md ../seaduck/docs/sciserver_notebooks/
```

6. Change directory back to seaduck. Follow step 4 to 7 in the [previous](text_file) section.
1. Before the changes are merged, check if the external links work by

```shell
make link-check
```

This check could have some persistent false positive, because some website don't like link checkers, which is indistinguishable from any other crawler. If you see a bad link, try it in your browser if it works that ignore the warning. If it still does not work, then find the proper link.

(deploy_doc)=

## Deploy documentation

https://jupyterbook.org/en/stable/start/publish.html#publish-your-book-online-with-github-pages
