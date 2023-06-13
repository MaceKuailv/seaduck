# Contributing to codes

I assume you already have a local git repository of the package. If you do not have that yet, follow [these instructions](use_git.md).

You should also have the environment ready, following [this tutorial](prep_env.md).

I can't tell you how to write good python code, because I am not great at all. However, [this](https://peps.python.org/pep-0020/) should be help.

After making your changes, make sure that you test the code with our existing tests powered by [pytest](https://docs.pytest.org),

```shell
make unit-tests
```

It is highly recommended, that you contribute to the `tests/` folder as well, especially if you just contributed new code to the package. The tests are the reason why people around the world can trust this package.

There are also some "style" requirements.

```shell
make qa
```

This will check the style of the code. Most of the times it will perform auto fixes.

When you are ready, [tidy up and make pull request](tidyNpr.md)
