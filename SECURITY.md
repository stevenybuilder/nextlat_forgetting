# Security and credential handling

This repository is public. Do not commit credentials, private keys, access
tokens, provider configuration containing secrets, private bucket URLs, or raw
cloud-runtime snapshots. Keep those values in environment variables or ignored
local files such as `.env` and `.secrets/`.

Before publishing a change:

1. Inspect the exact staged diff and list of staged files.
2. Run a secret scanner over the staged snapshot and the Git history.
3. Confirm that checkpoints, datasets, run caches, and local state remain
   ignored unless a specific artifact has been reviewed for public release.
4. Confirm that manifests expose hashes and reproducibility metadata, not
   credentials or signed download URLs.

If a credential is committed, revoke or rotate it immediately. Removing it in
a later commit is not sufficient because the value remains in Git history and
may already have been copied. After rotation, report the exposure through the
repository's private GitHub security-advisory channel rather than a public
issue.

Scientific artifacts may contain hostnames, usernames, absolute paths, cloud
object names, or job identifiers even when they contain no credential. Review
those fields separately for privacy before release.
