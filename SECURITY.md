# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x   | yes       |

## Reporting a vulnerability

Please report security issues privately via GitHub **Security Advisories →
"Report a vulnerability"** on this repository, or by email to the maintainers
at `[SECURITY CONTACT — to be filled by the repository owner]`.

Please do not open a public issue for a security report.

## Scope notes

- This package is a research/analysis tool that processes local files. It
  does not open network connections, and the released pipelines do not
  execute downloaded content.
- `torch.load` is used to read project-generated `.pt` files. Only load
  `.pt` files from sources you trust (this is standard for PyTorch-based
  software).
