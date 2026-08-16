#!/bin/bash
# Downloads real SIFT1M data. MUST be run on your own machine -- the
# sandbox that built this repo can't reach either host below (verified:
# both corpus-texmex.irisa.fr and huggingface.co return connection-blocked
# errors from that environment's network allowlist).
#
# Produces: sift1m/sift_base.fvecs, sift1m/sift_query.fvecs, sift1m/sift_groundtruth.ivecs
set -e

mkdir -p sift1m
cd sift1m

echo "== Option 1: original source (FAISS's own demo points here) =="
echo "If this hangs or fails (FTP is sometimes blocked by ISPs/firewalls):"
echo "  curl -o sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
echo "  tar -xzf sift.tar.gz --strip-components=1"
echo ""
echo "Attempting it now..."
if curl -f -o sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz 2>&1; then
    tar -xzf sift.tar.gz --strip-components=1
    rm sift.tar.gz
    echo "Downloaded via FTP."
else
    echo ""
    echo "FTP failed. == Option 2: HuggingFace mirror (HTTPS, more firewall-friendly) =="
    echo "Run manually:"
    echo "  pip install huggingface_hub"
    echo "  python3 -c \"from huggingface_hub import hf_hub_download as d; \\"
    echo "    print(d('qbo-odp/sift1m', 'sift_base.fvecs', repo_type='dataset')); \\"
    echo "    print(d('qbo-odp/sift1m', 'sift_query.fvecs', repo_type='dataset')); \\"
    echo "    print(d('qbo-odp/sift1m', 'sift_groundtruth.ivecs', repo_type='dataset'))\""
    echo "Then copy/symlink the printed paths to sift1m/sift_base.fvecs etc."
fi

echo ""
echo "Once files are in place, verify:"
echo "  cd .. && python3 -c \"from vecs_io import fvecs_read; x = fvecs_read('sift1m/sift_base.fvecs'); print(x.shape, x.dtype)\""
echo "Expected: (1000000, 128) float32"
