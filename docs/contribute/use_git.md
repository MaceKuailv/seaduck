# Using Git and Github

It essential for developer to use [Git](https://git-scm.com), especially for open source development. [GitHub](https://github.com) is a website based on the idea of git.

[GitHub](https://github.com)  not only hosts the source code, but also the documentation, tests, past issues, solutions, and the entire history of changes to the directory. Therefore, you need to have a [GitHub](https://github.com)  account.

Sign into the account, go to [seaduck](https://github.com/MaceKuailv/seaduck) and [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the directory by clicking around.

```{note} If you already have a fork, make sure you [sync](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork) them  before cloning it.
```

Almost ready. Now, let's get all the files from the repository first.

```shell
git clone https://github.com/YourGithubNickname/seaduck.git
```

Another option is to use the ssh link, which will make things more flexible

```shell
git clone git@github.com:YourGithubNickname/seaduck.git
```

If you use the second option, you will need to set up your github ssh key following this [tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).

To keep things tidy, it is recommended that you use a new branch of your fork to do the development. You can create that by running (from the `seaduck` folder):

```shell
git checkout -b name_of_your_new_branch
```

You can now make the changes you want to make in the repository. Checkout this [guide on contributing to code](contr_cod.md) and this [guide on contributing to documentation](contr_doc.md)for our recommended practices. Every once a while, run

```shell
git add .
git commit -m "Message describing your edits"
```

This will create a milestone to revert back to. For most developers, running this command also make them feel pretty good.

When you are done with all your changes, or when you just want to keep collaborators updated on what you have done, run

```shell
git push -u origin name_of_your_branch
```

If you go to your directory now, you will find out that you can make a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

When a pull request is created or when new changes are pushed to the pull request, some automatic changes will run. If the runs are not successful, github will send you loads of personal emails, which can be annoying. Therefore, I recommend [tidying up before pull request, following this tutorial](tidyNpr.md).
