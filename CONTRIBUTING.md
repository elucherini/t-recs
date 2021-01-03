# Contributing to T-RECS
Thank you for your interest in contributing to T-RECS! Please use the following guidelines on how to contribute to this project.

## Reporting issues
If you find a bug while using any tools in this repository, consider creating a [GitHub issue](https://github.com/elucherini/t-recs/issues) documenting the problem. Be sure to be detailed. Explain what you did, what you expected to happen, and what actually happened. You can also open a new issue if you'd like to request a feature. Please be sure to make use of the different labels that you can apply to issues (i.e., bug, enhancement, documentation, etc.)

## Writing new code

### Getting up and running with your dev environment
Before actually writing any code, you'll want to run `pip install -r requirements-dev.txt` from the root directory of this repository. The packages included will enable you to run unit tests and lint your code easily. Once you install those packages, try running `./scripts/lint_and_test.sh`. If that script finishes with no errors, you're good to go; otherwise, feel free to open up a new issue alerting us to what went wrong.

**N.B. (December 2020)** We are aware of some package conflicts between `pylint` and `pandas` that may result in a `MaximumRecursionError` when you run unit tests; if you have `pandas` in your environment, one fix is to up/downgrade `pandas` to `v1.0.5`. For more information, see the discussion [here](https://github.com/PyCQA/pylint/issues/3746).

### Updating documentation
If you find any documentation that is lacking or in error, submit a pull request that updates it.

### Contributing code via pull requests

The main way to contribute code to the project is via GitHub pull requests (PRs). To learn how to create a pull request, see the [GitHub docs](https://help.github.com/articles/creating-a-pull-request/). Be detailed in the description of your pull request. Explain what your proposed changes do, and reference any relevant issues. **Make sure that your PR passes the linter and the unit test suite**; otherwise, it will not be merged.

Here are some rules to follow for writing PRs:

* Ensure that an individual PR is relatively small. If the PR will be >80 lines of code, please open up an issue and discuss with a maintainer before starting work on the PR.
* Have easy-to-understand commit messages (e.g., follow the suggestions [here](https://chris.beams.io/posts/git-commit/)).
* Fix only one issue at a time (rather than fix multiple issues in one giant P.R.).
* Write unit tests for new changes (and ensure existing unit tests pass).
* Respond to any comments from reviewers.
* Comment on individual lines of the PR yourself if you want to bring special attention to it for feedback.
* In the case that a PR is "good enough for now" but it is known that it will induce future changes, document that foreseen problem as an Issue immediately.

If you're reviewing a PR:
* Don't accept commented out code.
* Don't accept functions with documentation.
* If you see code that you don't understand, make liberal use of comments to clarify the author's intent.
* If the person submitting the PR is a core contributor, do not merge the PR yourself, just approve it. The person submitting the PR should merge it.
* If the person submitting the PR is not a core contributor, the reviewer should merge the PR.

### Style guide
We follow the [PEP 8 style guide](https://www.python.org/dev/peps/pep-0008/) by way of the `black` auto-formatter tool. To run `black` on any file you've written, run `python -m black trecs/[path/to/file]`. We also use `pylint` to check for errors, like unused imports or variables.
