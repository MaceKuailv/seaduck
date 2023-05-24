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

Now that you have the repository, come inside by

```shell
cd seaduck
```

The package does not depend on a lot of the packages. If you have a conda environment that already have xarray installed, it is highly unlikely to have an environmental conflict. If you want to create a new conda environment anyway, run

```shell
conda create --name bubblebath
conda activate bubblebath
```

Install the basic dependencies. if you are running it from sciserver's Oceanography image, this is already satisfied, you can skip this step. Otherwise,

```
conda env update --file environment.yml
```

There is some extra dependencies for testing, run

```
conda env update --file ci/environment-ci.yml
```

If you are running with sciserver's Oceanography image, you still need to run the above step, unless you don't want to. 

You might be asking: "Hey! Is there a step that is absolutely not optional?" Yes, you need to install the package. 

```shell
pip install -e .
```

