build-convert-awq:
	DOCKER_BUILDKIT=1 docker build --ssh default -t convert-awq:latest -f Dockerfile .

tag-convert-awq:
	docker tag convert-awq:latest rekaai/convert-awq:latest

push-convert-awq:
	docker push rekaai/convert-awq:latest