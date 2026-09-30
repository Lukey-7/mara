"""Make Pydantic JSON schemas portable across LLM structured-output APIs.

Pydantic emits nested models as `$defs` + `$ref`. Provider support for references is uneven
(and changes between API versions), so we inline every `$ref` into a self-contained schema
and drop `title` keys, which only add noise to the prompt the provider builds from it.
"""

from typing import Any


def inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.get("$defs", {})

    def resolve(node: Any, seen: tuple[str, ...] = ()) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                if name in seen:  # recursive model: leave a plain object rather than loop
                    return {"type": "object"}
                return resolve(defs[name], (*seen, name))
            return {k: resolve(v, seen) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [resolve(v, seen) for v in node]
        return node

    return resolve(schema)
