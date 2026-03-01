# Security Notes

## Relevant Runtime Controls

Security-related controls are configured from JSON config files and enforced in API middleware.

### Config Areas

- `security.rate_limit`
- `security.auth`
- `security.cors`

See `config/base.json` and environment overrides in `config/`.

### API Protections

- Input/request size checks in diagnosis endpoint.
- Middleware-based request size limiting.

## Reporting Security Issues

Follow the repository policy in the root `SECURITY.md`.

## Operational Recommendation

- Keep production secrets out of committed JSON.
- Restrict CORS origins in production.
- Enable and tune rate limits for public endpoints.
