# Tidy it up and make Pull Request

Every time you make some change, run

```shell
git add .
git commit -m "A description of the good stuff you have done"
```

When you are in the `seaduck` directory and feel like you are finished with your changes, just before you submit everything, simply run

```shell
make
```

This will do the style check, run pytest. If you followed the steps in ["prepare the environment"](prep_env.md), everything here should pass. If this is the case, then

```shell
git push
```

Now open your forked directory, you should be able to "start a pull request" by clicking around.

Now, wait Mr. Duckmaster to merge your branch into the main branch.
