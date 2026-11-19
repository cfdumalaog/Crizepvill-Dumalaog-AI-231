# ME3 GitHub Pages deployment

ME3 can build a static website under `agent_workspace/<site-name>`, run its
local checks, preview it on the LAN, and publish it to GitHub Pages through the
`deploy_github_pages` tool. The voice agent never receives a GitHub token in a
prompt and never writes one to the repository, command line, tool event log, or
handoff.

The configured default output repository is
`cfdumalaog/Crizepvill-Dumalaog-AI-231` on `main`, with the Pages source at the
repository root. ME3 uses that target when the user asks for GitHub output
without naming another repository. It is only a convenience default, not an
allowlist: an explicitly named `owner/repository` is accepted when the private
GitHub credential can read and write it. A clear request to publish executes
without an extra approval exchange.

## Credential setup

Do not paste a GitHub token into chat or a voice request. Use the Pi UI through
the existing local SSH tunnel, then open the **GitHub Pages** setup card and
choose **Validate & save GitHub token**. The endpoint accepts credential writes
only from a loopback request. On Linux the token is stored in the ME3 service
user's private `~/.config/me3/credentials.json` with mode `0600`; an environment
variable `GITHUB_TOKEN` may be used instead for a managed deployment.

For access to every repository that the account is allowed to manage, create a
fine-grained token with **Repository access: All repositories**. Give it
**Metadata: Read-only**, **Contents: Read and write**, and **Pages: Read and
write** (and the relevant organization approval, if GitHub requires it). The
ME3 read-only discovery and access tools show the permissions that GitHub
actually returns; code cannot expand a token's scope. A token limited to one
repository remains valid, but ME3 will correctly expose only that repository
and reject a deployment to a target it cannot write.

The desktop GitHub connector is not copied into the Pi. It is a separate
operator connection and does not expose its credential to the ME3 process.

## Voice workflow

1. Ask ME3 to build the site, for example: “Build a responsive portfolio site
   under `agent_workspace/portfolio` and verify that it loads.”
2. Review the LAN preview and ask for any edits.
3. Use the configured default target or provide another target, for example:
   “Deploy the validated `agent_workspace/portfolio` site to
   `alice/portfolio-site` on `main`.”
4. ME3 checks the selected repository and, if the user requested a new one,
   creates it with an initialized branch. The GitHub credential must permit
   repository creation for this step.
5. ME3 creates one atomic Git commit through the GitHub
   Git-data API, verifies the branch update, and checks or enables GitHub
   Pages. It returns the commit identifier and Pages URL without returning the
   token.

The site directory must remain under `agent_workspace`, contain `index.html`,
and stay within the bounded file-count and upload-size limits in
`me3/github_pages.py`. `github_repositories` lists visible repositories and
`github_repository_access` checks one named target without modifying it. The
publisher updates or adds site files but does not delete unrelated repository
files.

## Direct tool parameters

```json
{
  "site_dir": "agent_workspace/portfolio",
  "owner": "github-owner",
  "repo": "github-owner.github.io",
  "branch": "main",
  "pages_path": "/",
  "message": "Deploy portfolio site from ME3",
  "enable_pages": true,
  "create_if_missing": true
}
```

For a project Pages repository, the expected URL is normally
`https://OWNER.github.io/REPOSITORY/`. For a user Pages repository named
`OWNER.github.io`, it is normally `https://OWNER.github.io/`.

## Safety and failure behavior

- A missing repository is created only when the user requests a new repository
  or sets `create_if_missing=true` in a direct tool call.
- Missing or rejected credentials are reported only as status names.
- A branch update conflict is reported and is not force-pushed.
- If the commit succeeds but Pages cannot be enabled, ME3 reports a partial
  commit and does not claim that the site is live.
- The current tool does not delete remote files, rewrite history, or use shell
  `git push`.
