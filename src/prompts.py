# Prompt format shared between training and production.
from __future__ import annotations
from .catalog import format_precio

NO_ENCONTRADO = "No encuentro ese producto en el catálogo actual."

SYSTEM_PROMPT = (
    "Eres el asistente de ventas de {tienda}. Responde en español, breve y amable, "
    "usando SOLO la información del CONTEXTO. "
    f'Si el producto no aparece en el contexto, responde: "{NO_ENCONTRADO}" '
    "No inventes precios, stock ni características."
)

def ficha(p) -> str:
  
    line = (
        f"- {p['nombre']} | categoría: {p['categoria']} | "
        f"precio: {format_precio(p['precio'])} Bs | stock: {int(p['stock'])}"
    )
    specs = str(p.get("specs") or "").strip()
    return line + (f" | {specs}" if specs else "")

def build_context(products) -> str:
    if not products:
        return "CONTEXTO:\n(sin resultados)"
    return "CONTEXTO:\n" + "\n".join(ficha(p) for p in products)


def build_user_prompt(products, question: str) -> str:
    return f"{build_context(products)}\n\nPREGUNTA: {question}"


def build_messages(tienda: str, products, question: str, answer: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT.format(tienda=tienda)},
        {"role": "user", "content": build_user_prompt(products, question)},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs