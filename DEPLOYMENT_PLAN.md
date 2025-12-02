# SQL Lineage - Deployment & Project Structure Plan

This document outlines the complete plan for extracting the SQL parser as a standalone library and restructuring the project for deployment.

## Current Status ✓

### Phase 1: Parser Library Extraction (COMPLETED)

The SQL parser has been successfully extracted into a standalone Python library:

**Location**: `/Users/mingjerli/repo/clgraph/clgraph/`

**Structure**:
```
clgraph/
├── .github/
│   └── workflows/
│       └── test.yml           # CI/CD for testing
├── src/
│   └── clgraph/
│       ├── __init__.py        # Public API exports
│       └── parser.py          # Core parser logic (2259 lines)
├── tests/
│   ├── __init__.py
│   ├── test_lineage.py        # Single-query tests
│   └── test_multi_query.py    # Pipeline tests
├── examples/
│   ├── simple_example.py      # Basic usage example
│   └── pipeline_example.py    # Multi-query example
├── pyproject.toml             # Modern Python packaging
├── README.md                  # Comprehensive documentation
├── LICENSE                    # MIT License
├── CONTRIBUTING.md            # Contribution guidelines
├── MANIFEST.in                # Package manifest
└── .gitignore

```

**Key Features**:
- ✓ Proper Python package structure with `src/` layout
- ✓ Modern packaging with `pyproject.toml` and hatchling
- ✓ Comprehensive test suite (56+ tests)
- ✓ GitHub Actions CI/CD workflow
- ✓ Development dependencies (pytest, black, ruff, mypy)
- ✓ MIT License
- ✓ Full documentation and examples

**Installation**:
```bash
# For development
uv venv
uv pip install -e ".[dev]"

# For production
pip install clgraph
```

**Testing**:
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_lineage.py -v

# Check formatting
black --check src/ tests/

# Lint
ruff check src/ tests/
```

## Next Steps

### Phase 2: Restructure clvisualize (TODO)

Make `clvisualize` a standalone repository with:

**New Structure**:
```
clvisualize/
├── .github/
│   └── workflows/
│       └── deploy.yml         # Deploy to VM
├── apps/
│   ├── lineage/               # Column lineage app
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   └── requirements.txt
│   └── pipeline/              # Pipeline lineage app
│       ├── Dockerfile
│       ├── app.py
│       └── requirements.txt
├── nginx/
│   ├── nginx.conf             # Reverse proxy config
│   └── Dockerfile
├── docker-compose.yml         # Orchestrate all services
├── .env.example
└── README.md
```

**Key Changes**:
1. Remove `sql_parser.py` - use `clgraph` from PyPI
2. Each app in its own Docker container
3. Nginx for reverse proxy and SSL
4. Single docker-compose for local dev and deployment

### Phase 3: Company Website (TODO)

Create a separate repository for marketing/documentation:

**Suggested Structure**:
```
website/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── src/
│   ├── pages/
│   │   ├── index.tsx          # Landing page
│   │   ├── blog/
│   │   ├── docs/
│   │   ├── about.tsx
│   │   └── demo.tsx           # Embed apps via iframe
│   ├── components/
│   └── styles/
├── public/
├── Dockerfile
├── package.json
└── README.md
```

**Technology Stack Options**:
- Next.js (React) - Modern, SEO-friendly
- Hugo (Static) - Fast, simple, Go-based
- Docusaurus - Documentation-focused

## Deployment Architecture

### Docker Compose Setup (Recommended for MVP)

```yaml
# docker-compose.yml
services:
  # Company website
  website:
    build: ./website
    container_name: company-website
    environment:
      - NODE_ENV=production
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.website.rule=Host(`yourcompany.com`)"

  # Column lineage demo
  app-lineage:
    build: ./clvisualize/apps/lineage
    container_name: demo-lineage
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.lineage.rule=Host(`demo.yourcompany.com`) && PathPrefix(`/lineage`)"

  # Pipeline lineage demo
  app-pipeline:
    build: ./clvisualize/apps/pipeline
    container_name: demo-pipeline
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.pipeline.rule=Host(`demo.yourcompany.com`) && PathPrefix(`/pipeline`)"

  # Reverse proxy with SSL
  traefik:
    image: traefik:v2.10
    container_name: reverse-proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./traefik:/etc/traefik
      - ./letsencrypt:/letsencrypt
```

### VM Deployment

**Recommended Provider**: DigitalOcean, AWS EC2, or Google Compute Engine

**Minimum Specs**:
- 2 vCPUs
- 4GB RAM
- 50GB SSD
- Ubuntu 22.04 LTS

**Setup Steps**:
```bash
# 1. Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 2. Clone repositories
git clone https://github.com/yourusername/website.git
git clone https://github.com/yourusername/clvisualize.git

# 3. Configure environment
cp .env.example .env
# Edit .env with production values

# 4. Start services
docker-compose up -d

# 5. Setup SSL (with Let's Encrypt)
# Traefik will automatically handle SSL certificates
```

## Repository Strategy

### Option 1: Monorepo (Simpler)
```
company-platform/
├── clgraph/          # Python library
├── clvisualize/         # Streamlit apps
├── website/             # Marketing site
└── docker-compose.yml   # Deploy all together
```

**Pros**: Single source of truth, easier coordination
**Cons**: Larger repo, slower CI, more coupling

### Option 2: Multi-repo (Recommended)
```
clgraph/              # Separate repo, PyPI package
clvisualize/             # Separate repo, depends on clgraph
company-website/         # Separate repo, embeds demos
```

**Pros**: Clear separation, independent deployments, easier to open-source clgraph
**Cons**: More repos to manage, need to coordinate releases

## Publishing Strategy

### SQL Lineage Library
1. Push to GitHub: `github.com/yourcompany/clgraph`
2. Publish to PyPI: `pip install clgraph`
3. Versioning: Semantic versioning (0.1.0 → 0.2.0 → 1.0.0)
4. Documentation: ReadTheDocs or GitHub Pages

### Demo Apps
1. Push to GitHub: `github.com/yourcompany/clvisualize`
2. Docker images: Push to Docker Hub or GitHub Container Registry
3. Deployment: Auto-deploy on push to main via GitHub Actions

### Website
1. Push to GitHub: `github.com/yourcompany/website`
2. Deployment: Auto-deploy via Vercel, Netlify, or self-hosted

## Timeline Estimate

- ✓ **Week 1**: Extract parser library (DONE)
- **Week 2**: Restructure clvisualize, Docker setup
- **Week 3**: Create company website (basic)
- **Week 4**: Setup VM, deploy everything, test
- **Week 5**: Polish, monitoring, documentation
- **Week 6**: Launch 🚀

## Cost Estimate (Monthly)

- VM (DigitalOcean Droplet): $24/month (4GB, 2 vCPUs)
- Domain name: $12/year
- SSL: Free (Let's Encrypt)
- GitHub: Free (public repos)
- **Total: ~$25/month**

## Next Actions

1. [ ] Initialize git repo for clgraph
2. [ ] Create GitHub repository
3. [ ] Push code and create first release
4. [ ] Restructure clvisualize as described above
5. [ ] Create company website skeleton
6. [ ] Setup development docker-compose
7. [ ] Test full stack locally
8. [ ] Provision VM
9. [ ] Deploy to production
10. [ ] Setup monitoring (optional: Grafana + Prometheus)

## Notes

- Keep clgraph library focused - no UI dependencies
- Demo apps should be lightweight and fast
- Consider adding analytics to website (Google Analytics, Plausible)
- Plan for scaling: Can move to Kubernetes later if needed
- Consider CDN for static assets (CloudFlare)
