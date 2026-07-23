# Contributing to documentation🦆

We are using [MyST](https://mystmd.org)📙 (the engine behind Jupyter Book 2) to build the documentation. The online documentation is hosted by [github pages](https://pages.github.com/).

## Add/Modify

MyST supports several different kinds of files as input, including [Markdown (.md files)](https://mystmd.org/guide/quickstart)⬇️, [reStructuredText (.rst files)](https://docutils.sourceforge.io/docs/user/rst/cheatsheet.html)📜, and [Jupyter Notebooks (.ipynb files)](https://jupyter.org/)🐍. Notebooks are executed at build time to generate outputs. Here are instructions on how to work with each type.

(text_file)=

## Text files

Adding files (markdown, reStructuredText, etc) are not very different from contributing code.

1. [Clone the git repository to your local machine](./use_git)
1. [Build the environment](prep_env.md). If you do not run this step, the API references and notebooks cannot be properly built.
1. Add or change the files. A useful cheat sheet can be found here: [MyST Markdown](https://mystmd.org/guide/quickstart), [reStructuredText](https://docutils.sourceforge.io/docs/user/rst/cheatsheet.html). [Commit](use_git.md) the changes you did along the way.
1. If you added a new file, add it to `docs/myst.yml` in the appropriate location within the table of contents structure. The file should include a `file:` entry with the relative path to your markdown or notebook file.
1. Change directory to seaduck, and run

```none
make docs-build
```

6. Run `make docs-serve` to start a live preview server at `http://localhost:3000`. This automatically rebuilds and reloads your changes in the browser whenever you edit files. Perfect for iterative development! If you see issues, edit and save—the page refreshes automatically.

7. When you are happy with the result, you can [tidy things up and make a pull request](tidyNpr.md). After approval, your changes will be {ref}`deployed <deploy_doc>`.

## Notebooks that could be run any where

As a package that works with oceanographic datasets, almost all the demonstrations requires some supporting data. `seaduck` provides several datasets available to be downloaded everywhere. It can be accessed simply by

```python
ds = seaduck.utils.get_dataset(name)
```

Another option is to generate the dataset using mathematical expressions out of thin air. Since this kind of notebook is executed whenever the github action is triggered, it is preferrable that these notebooks run very fast. For example, do not perform heavy calculations in those notebooks and please do not install packages within them.

The procedure is similar to [text files](#text_file), with one additional step:

1. Follow steps 1-4 in the [previous](#text_file) section.
2. Run

```bash
make qa
```

This step runs code quality checks with `pre-commit` and strips notebook outputs (keeping notebooks clean in git). If it fails the first time, pre-commit will automatically fix most issues. Run it again - it should pass on the second attempt. If errors persist, check the error messages.

3. The notebooks will be automatically executed at build time (`make docs-build`) with outputs embedded in the HTML.
4. [Tidy things up and make a pull request](tidyNpr.md)

## Cooler (Sciserver) notebooks

Cool stuff are not always portable. The ocean 🌊 is an example of that.
Say you have something really cool you want to demonstrate, but the dataset it is based on is to large to distribute or it simply takes too long to run. Wouldn't it be nice if we could have a cloud platform that host a bunch of ocean dataset that is free for everyone to use? It would be even better if the packages I need as an oceanographer is readily installed and I don't have to worry about a thing.
You can use [Sciserver](https://sciserver.org/)! (Am I too dramatic?). Sciserver is also the home base of [oceanspy](https://oceanspy.readthedocs.io/en/latest/), a package that will make your life so much easier as a oceanographer. After registering on sciserver (you can follow this youtube tutorial [here](https://www.youtube.com/channel/UCpYkjUrm2a_ANY86Fb4uyvg)), you can simply call this oceanspy function

```python
import ocenspy as ospy

od = ospy.open_oceandataset.from_catalog("NameOfDataset")
ds = od._ds
```

Note that since you are using the `Oceanography` image on SciServer, most packages are already installed. You only need one dependency to convert notebooks to markdown:

```bash
pip install jupytext
```

That's it—you don't need to set up the local environment.

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

2. Create and execute your notebooks. **Important**: Always use `plt.show()` after plots so they render correctly.

3. In `seaduck_sciserver_notebook`, run the conversion script

```bash
python convert_ipynb.py
```

This script converts `.ipynb` files to `.md` files and adds metadata about when the notebook was last run. The script looks for your name in the file and adds a timestamp note on the next line. If you're not the original author, just add a comment with your name.

4. The generated markdown files will reference plots via GitHub URLs (e.g., `https://github.com/MaceKuailv/seaduck_sciserver_notebook/blob/master/notebook_files/plot.png?raw=true`). These links will only work once the markdown files are merged to the main branch. Commit your changes and make a pull request to the `seaduck_sciserver_notebook` repo. Once merged to `main`, verify the pages render correctly in your browser.

5. Copy the generated markdown files to seaduck:

```bash
cp *.md ../seaduck/docs/sciserver_notebooks/
```

6. Go back to the seaduck directory and run `make docs-build` to verify the pages build correctly. Use `make docs-serve` for interactive preview.

7. Before the changes are merged, check if the external links work by running:

```bash
make link-check
```

This check could have some persistent false positive, because some website don't like link checkers, which is indistinguishable from any other crawler. If you see a bad link, try it in your browser if it works that ignore the warning. If it still does not work, then find the proper link.

(deploy_doc)=

## Deploy documentation

The documentation is automatically deployed to GitHub Pages when changes are merged to the main branch via CI/CD. The static HTML files in `docs/_build/html/` are built with `make docs-build` and deployed by GitHub Actions. 

For more details on manual deployment or customizing the deployment process, see:
- [GitHub Pages documentation](https://pages.github.com/)
- [MyST deployment guide](https://mystmd.org/guide/publishing)
