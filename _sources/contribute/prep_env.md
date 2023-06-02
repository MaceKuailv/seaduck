# Prepare the environment

First fork a copy of the package. Let's get all the files from the repository first.

```shell
git clone https://github.com/YourGithubNickname/seaduck.git
```

Another option is to use the ssh link, which will make things more flexible

```shell
git clone git@github.com:YourGithubNickname/seaduck.git
```

If you use the second option, you will need to set up your github ssh key following this [tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

The above will probably go into the use_git.md file.

Now that you have the repository, come inside and make your self comfortable. Run

```shell
cd seaduck
```

```{tip}
If you are running with sciserver's Oceanography image or your own environment that already have the basic packages installed, you pretty much just need to run the following to get all the dependencies:
`conda env update --file ci/environment-ci.yml`
```

If you want to create a new conda environment, run

```shell
conda create --name bubblebath
conda activate bubblebath
```

I recommend installing the basic dependencies and the extra dependencies for tests at the same time. First, we create a file that combines the two `yml` file by running

```shell
pip install conda-merge
conda-merge environment.yml ci/environment-ci.yml > PleaseIgnore_env.yml
```

Now, let's install them

```
conda env update --file environment.yml
```

Working with environment could be a headache sometimes. If some of the packages is unable to install on your machine, don't sweat over it. Skip them for now, it is totally possible that some packages is not necessary for your purposes.
You might be asking: "Hey! Is there a step that is absolutely not optional?" Yes, you need to install our package if you want to work on it.

```shell
pip install -e .
```
