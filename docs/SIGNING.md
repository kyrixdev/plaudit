# Code signing with SignPath

Plaudit's Windows builds are signed through the [SignPath Foundation](https://signpath.org) programme, which gives
free code signing to open-source projects. Signing removes "Unknown Publisher" from Windows' warnings. SmartScreen's
"Windows protected your PC" screen fades as the signed files build up a download reputation.

SignPath only signs files built by **GitHub Actions on GitHub-hosted runners** from this public repository, so it can
verify that each binary comes from this source code. The workflow is
[`.github/workflows/release.yml`](../.github/workflows/release.yml):

1. Runs the tests and builds `Plaudit.exe` (`build.py --app-only`).
2. Sends the app folder to SignPath, which signs **only `Plaudit.exe`**. The bundled third-party runtime files are
   left as they are, as the Foundation's rules require.
3. Builds the installer around the signed app (`build.py --setup-only`), so installed copies are signed too.
4. Sends the installer to SignPath and signs it.
5. Checks both signatures, uploads the installer as a workflow artifact, and can create a draft GitHub release.

With the signing policy set to `none`, the same workflow produces an unsigned build. That's useful before SignPath is
set up.

## One-time setup

1. **Apply** at <https://signpath.org/apply>. The Foundation's [terms](https://signpath.org/terms) require, among
   other things:
   - an OSI-approved licence (MIT ✓);
   - an uninstaller (✓);
   - a published "Code signing policy" (the [README section](../README.md#code-signing-policy) ✓);
   - multi-factor authentication on GitHub and SignPath for everyone on the team;
   - a manual approval for every release.
2. **In SignPath**, once your organisation exists:
   - Add the predefined Trusted Build System **GitHub.com** to the organisation.
   - Create a project with the slug **`plaudit`**, linked to `https://github.com/kyrixdev/plaudit`, and link the
     GitHub.com trusted build system to it.
   - Add two artifact configurations, pasting in the files from
     [`.signpath/artifact-configurations/`](../.signpath/artifact-configurations/):
     - slug **`app`**: `app.xml`
     - slug **`installer`**: `installer.xml`

     Both require the product name `Plaudit` and the product version `${version}`. The build sets these in the files'
     version resources, and the workflow passes the version from `plaudit/__init__.py`.
   - Use the signing policies **`test-signing`** and **`release-signing`**. On `release-signing`, turn on
     **Verify origin** and set *Allowed branch names* to `main`.
   - Create a CI user with the *Submitter* role on the policies, and generate an API token for it.
3. **In GitHub** (repository **Settings → Secrets and variables → Actions**):
   - Secret **`SIGNPATH_API_TOKEN`**: the CI user's API token.
   - Variable **`SIGNPATH_ORGANIZATION_ID`**: your SignPath organisation ID.
   - Variable **`SIGNPATH_PROJECT_SLUG`**: only needed if your project slug isn't `plaudit`.

## Making a signed release

1. Bump `__version__` in `plaudit/__init__.py`, commit and push to `main`.
2. Go to **Actions → Release build → Run workflow** on `main`. Choose `release-signing`, and tick
   *Create a draft GitHub release*.
3. Approve the two signing requests in SignPath (`Plaudit.exe`, then the installer). The workflow waits up to an hour
   for each one.
4. Review the draft release on GitHub and publish it.

Try `test-signing` first. It runs the whole pipeline with SignPath's test certificate.
