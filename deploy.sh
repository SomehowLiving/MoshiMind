set -ex

# Override for forks, e.g.: DEPLOY_HOST=my-server.example.com ./deploy.sh
DEPLOY_HOST="${DEPLOY_HOST:-moshi-rag.kyutai.org}"

rm -rf ./moshi/.venv ./moshi/dist ./rust/target
docker compose -f ./swarm-config.yaml build  --push --progress=plain

docker -H "ssh://root@${DEPLOY_HOST}" stack deploy -c ./swarm-config.yaml --with-registry-auth --prune moshi-rag
