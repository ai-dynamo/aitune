---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Contributing"
---

Contributions are welcome, and they are much appreciated! Every little
helps, and we will always give credit.

## Types of Contributions

### Report Bugs

Report bugs at [https://github.com/ai-dynamo/aitune/issues](https://github.com/ai-dynamo/aitune/issues).

If you are reporting a bug, include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement" and "help wanted" is open to whoever would like to implement it.

### Write Documentation

NVIDIA AITune could always use more documentation, whether as part of
the official NVIDIA AITune docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at [https://github.com/ai-dynamo/aitune/issues](https://github.com/ai-dynamo/aitune/issues).

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible to make it easier to implement.

## Sign your Work

We require that all contributors "sign-off" on their commits. This certifies that
the contribution is your original work, or you have the rights to submit it under
the same license or a compatible license.

**Note:** Organization members are exempt from the DCO requirement; their commits
do not need to be signed off.

Any contribution from non-members which contains commits that are not Signed-Off
will not be accepted.

To sign off on a commit, you simply use the `--signoff` (or `-s`) option when committing your changes:
```
$ git commit -s -m "Add cool feature.
```

This will append the following to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

By doing this, you certify the below:

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
```

## Get Started!

Ready to contribute? Here's how to set up the `NVIDIA AITune` for local development.

1. Fork the `NVIDIA AITune` repo on GitHub.
2. Clone your fork locally:

```shell
$ git clone git@github.com:your_name_here/aitune.git
```

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development:

```shell
$ mkvirtualenv aitune
$ cd aitune/
$ make install-dev
```

4. Create a branch for local development:

```shell
$ git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

5. When you're done making changes, check that your changes pass linters and the
   tests, including testing other Python versions with tox:

```shell
$ make lint  # will run i.a. flake8 and pytype linters
$ make test  # will run a test with on your current virtualenv
$ make test-fw  # will run a framework test inside framework container
```

6. Commit your changes and push your branch to GitHub:

```shell
$ git add .
$ git commit -s -m "Your detailed description of your changes."
$ git push origin name-of-your-bugfix-or-feature
```

7. Submit a pull request through the GitHub website.

### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, you should update the docs. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.

## Documentation

Add/update docstrings as defined in the [Google Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).

