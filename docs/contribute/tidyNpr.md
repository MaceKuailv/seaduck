# Tidy it up and make Pull Request

When you are in the `seaduck` directory and feel like you are finished with your changes, just before you submit everything, simply run

```shell
make
```

This will do the style check and run `pytest`. If you followed the steps in ["prepare the environment"](prep_env.md), everything here should pass.

Make sure before you push everything is committed.

```shell
git add .
git commit -m "A description of the good stuff you have done"
```

````
```{tip} Sometimes the style check will make changes to the file, make sure you commit after that.
Alternatively, you could ask `pre-commit` to run every time you commit by running `pre-commit install`.
```
````

If everything passed then

```shell
git push -u origin name_of_your_branch
```

Now open your forked directory, you should be able to "start a pull request" by clicking around.

Now, wait Mr. Duckmaster to merge your branch into the main branch.
