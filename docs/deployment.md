# Static dashboard deployment

Live URL: <https://ukp-peer-review-pipeline.vercel.app>

The dashboard is a static Vercel project named `ukp-peer-review-pipeline`, owned by the `harshjoshi23s-projects` scope. It was deployed from a separate local hosting directory using Vercel CLI 64.5.0. No Git commit, push, or GitHub connection is needed for this route. Existing Vercel projects are unaffected.

## What is published

The homepage includes an interactive explorer, the optional model demonstration, and download links. Downloads include the required two programming files, the HLD and analysis report, JSON/pickle model artifacts, metrics, diagnostics, and a source ZIP containing the supporting code, notebook, and documentation.

The page embeds derived review records and anomaly explanations. It is publicly readable, not an aggregate-only page. The raw CSV, assessment PDF, application user ID, correspondence, and Part 1 materials are not in the hosting package. No credentials, local environment files, or Git metadata are served. The `noindex, nofollow` header is an indexing preference, not a privacy barrier.

## Rebuild the static bundle

From the programming repository root:

```bash
python3 tools/package_site.py ../ukp-dashboard-site
```

This packages the already-generated dashboard and artifacts. If the dataset or model changes, first follow the rebuild steps in [extras/README.md](../extras/README.md); packaging does not retrain the model. Keep the hosting directory outside this repository.

The separate hosting directory uses this `vercel.json` configuration:

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "framework": null,
  "buildCommand": "",
  "installCommand": "",
  "outputDirectory": "dist",
  "headers": [{
    "source": "/(.*)",
    "headers": [
      {"key": "X-Robots-Tag", "value": "noindex, nofollow"},
      {"key": "X-Content-Type-Options", "value": "nosniff"},
      {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"}
    ]
  }]
}
```

Its `.vercelignore` excludes `.openai/`, `.git/`, `.vercel/`, `.env`, `.env.*`, and `bundle-checksums.json`. Vercel serves only `dist`. There is no runtime Python server, dependency installation, or build command.

## Redeploy without Git

These commands use the existing project. They require Node.js 20+ and the correct Vercel account. Run from the programming repository root:

```bash
npm exec --yes --package vercel@64.5.0 -- vercel whoami
```

Confirm that the printed username is `harshjoshi23` before continuing:

```bash
npm exec --yes --package vercel@64.5.0 -- vercel deploy \
  --cwd ../ukp-dashboard-site --prod --yes \
  --scope harshjoshi23s-projects --project ukp-peer-review-pipeline
```

Use the stable production URL for sharing. A deployment-specific preview URL may require Vercel authentication even when the production URL is public. Verify the production URL in a signed-out browser and check the downloads after each deployment. [Vercel's deployment documentation](https://vercel.com/docs/cli/deploy) describes the CLI workflow.

Hosting and GitHub publication do not submit the assessment. Upload `solution.py` and `results.json` through the UKP programming form, and the completed review separately through its paper-review form.
