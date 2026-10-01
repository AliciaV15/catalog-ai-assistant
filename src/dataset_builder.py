"""Generador del dataset de fine-tuning (comportamiento del asistente).

Parte 2A: preguntas de DESCUBRIMIENTO (categoría, presupuesto, atributo, marca,
"el más barato") y preguntas por un PRODUCTO concreto (ficha, precio, stock, specs).
"""
from __future__ import annotations

import random
import unicodedata

import pandas as pd

from .catalog import format_precio
from .prompts import build_messages

# =====================================================================
# 1. BANCOS DE PREGUNTAS
#    Q_TRAIN se usa para entrenar. Q_TEST son formas de preguntar DISTINTAS
#    que el modelo nunca verá en train: sirven para medir si generaliza.
#    Variables: {n}=producto  {c}=categoría  {b}=presupuesto  {a}=atributo  {m}=marca
# =====================================================================
Q_TRAIN = {
    # --- descubrimiento ---
    "cat_budget": [
        "¿Qué {c} tienen que cuesten menos de {b} Bs?", "{c} hasta {b} Bs",
        "Busco {c} que no pase de {b}", "¿Tienen {c} de menos de {b}?",
        "Tengo {b} Bs, ¿qué {c} me recomiendan?", "algún {c} barato? máximo {b}",
        "¿Qué {c} hay por {b} Bs o menos?", "Necesito {c}, mi presupuesto es {b} Bs",
        "{c} máximo {b}",
    ],
    "cat_only": [
        "¿Qué {c} tienen?", "Quiero ver {c}", "¿Tienen {c}?", "Busco {c}",
        "Mándame opciones de {c}", "{c}?", "¿Qué {c} venden?", "Hola, ¿qué {c} manejan?",
    ],
    "cat_cheapest": [
        "¿Cuál es la opción más barata en {c}?", "Lo más económico en {c}",
        "¿Qué {c} es lo más barato que tienen?", "{c} más baratos?",
        "Busco {c}, lo más barato que haya",
    ],
    "cat_attr": [
        "{c} con {a}", "¿Tienen {c} que tengan {a}?", "Busco {c} con {a}",
        "¿Qué {c} traen {a}?", "Necesito {c} con {a}, ¿hay?", "{c} {a}",
    ],
    "marca": [
        "¿Qué tienen de {m}?", "¿Tienen algo de {m}?", "Productos {m}",
        "¿Qué hay de {m}?", "algo de {m} porfa",
    ],
    # --- producto concreto ---
    "ficha": [
        "¿Tienen {n}?", "Hola, busco {n}", "¿{n} disponible?", "Buenas, ¿tienen {n}?",
        "Info de {n} porfa", "Me interesa {n}, ¿lo tienen?", "¿Tienen {n} nomás?", "{n}?",
    ],
    "precio": ["¿Cuánto cuesta {n}?", "Precio de {n}", "¿A cuánto está {n}?", "¿Cuánto sale {n}?", "cuanto es {n}"],
    "stock": ["¿Cuántas unidades tienen de {n}?", "¿Hay stock de {n}?", "¿Les queda {n}?", "¿Todavía tienen {n}?"],
    "specs": [
        "¿Qué características tiene {n}?", "¿Qué specs tiene {n}?", "¿Qué trae {n}?",
        "Cuéntame qué tiene {n}", "Detalles técnicos de {n}",
    ],
}

Q_TEST = {
    "cat_budget": ["Mi plata llega a {b} Bs, ¿qué {c} hay?", "¿Hay {c} por debajo de {b}?", "{c} que cuesten {b} o menos"],
    "cat_only": ["Dame {c}", "¿Con qué {c} cuentan?", "Estoy buscando {c}"],
    "cat_cheapest": ["¿Cuál sale más barato en {c}?", "{c} más económicos que haya"],
    "cat_attr": ["Quiero {c} que tengan {a}", "¿Hay {c} con {a}?", "{c} que traiga {a}"],
    "marca": ["¿Manejan {m}?", "¿Qué modelos de {m} hay?"],
    "ficha": ["¿Cuentan con {n}?", "Quisiera saber si hay {n}", "{n} ¿lo tienen disponible?"],
    "precio": ["¿Me dice el precio de {n}?", "¿Cuál es el valor de {n}?"],
    "stock": ["¿Cuántos {n} quedan?", "¿Tienen unidades de {n}?"],
    "specs": ["¿Qué lleva {n}?", "¿Cómo es {n}? Quiero saber sus especificaciones"],
}

# Cómo la gente llama a cada categoría de TU catálogo. Si una categoría no está
# aquí, se usa su nombre en minúscula. ✏️ Edítalo con las categorías de tu Sheet.
SINONIMOS = {
    "Laptops": ["computadoras", "computadora", "compu", "laptops", "laptop", "notebook", "notebooks"],
    "Celulares": ["celulares", "celular", "teléfonos", "teléfono", "smartphones", "smartphone"],
    "Monitores": ["monitores", "monitor", "pantallas"],
    "Audífonos": ["audífonos", "auriculares", "audífono", "headsets"],
    "Accesorios": ["accesorios"],
    "Impresoras": ["impresoras", "impresora"],
}


# =====================================================================
# 2. RUIDO: para que las preguntas suenen humanas
# =====================================================================
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _noisy(q: str, rng: random.Random) -> str:
    if rng.random() < 0.25:
        q = _strip_accents(q).replace("¿", "").replace("¡", "")
    if rng.random() < 0.15:
        q = q.lower()
    if rng.random() < 0.10:
        q = q.replace("?", "")
    return q.strip()


def _typo(s: str, rng: random.Random) -> str:
    """Intercambia dos letras vecinas: 'IdeaPad' -> 'IdePaad'."""
    if len(s) < 6:
        return s
    i = rng.randrange(1, len(s) - 2)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:]


def _cat_word(cat: str, rng: random.Random) -> str:
    return rng.choice(SINONIMOS.get(cat, [cat.lower()]))


def _budget_text(b: int, rng: random.Random) -> str:
    """5000 -> '5000', '5.000', '5 mil', '5mil' o '5k'."""
    opts = [str(b), str(b), format_precio(b)]
    if b % 1000 == 0:
        opts += [f"{b // 1000} mil", f"{b // 1000}mil", f"{b // 1000}k"]
    return rng.choice(opts)


def _is_unique(fragment: str, records: list[dict]) -> bool:
    """¿El fragmento identifica a UN solo producto? (evita ambigüedades)"""
    frag = fragment.lower()
    return sum(frag in r["nombre"].lower() for r in records) == 1


def _mention(p: dict, records: list[dict], rng: random.Random) -> str:
    """Cómo nombra el cliente al producto: completo, sin marca, parcial o con typo."""
    name, brand = p["nombre"], str(p.get("marca") or "")
    cands = [name.lower()]
    if brand and name.lower().startswith(brand.lower()):
        cands.append(name[len(brand):].strip())      # "IdeaPad 3"
    words = name.split()
    if len(words) > 2:
        cands.append(" ".join(words[:2]))            # "Lenovo IdeaPad"
    opts = [name, name] + [c for c in cands if c and _is_unique(c, records)]
    m = rng.choice(opts)
    return _typo(m, rng) if rng.random() < 0.10 else m


def _tokens(specs: str) -> list[str]:
    return [t.strip().lower() for t in str(specs).split(",") if t.strip()]


# =====================================================================
# 3. RESPUESTAS: SIEMPRE construidas con los datos del contexto
# =====================================================================
def _unidades(s: int) -> str:
    return "1 unidad disponible" if s == 1 else f"{s} unidades disponibles"


def a_ficha(p: dict) -> str:
    n, s, precio = p["nombre"], int(p["stock"]), format_precio(p["precio"])
    if s == 0:
        return f"Por ahora {n} está agotado 😕. Su precio de referencia es {precio} Bs."
    verbo = "nos queda" if s == 1 else "tenemos"
    txt = f"Sí 😊 {n} está disponible. Actualmente cuesta {precio} Bs y {verbo} {_unidades(s)}."
    specs = str(p.get("specs") or "").strip()
    return txt + (f" Características: {specs}." if specs else "")


def a_precio(p: dict) -> str:
    precio = format_precio(p["precio"])
    if int(p["stock"]) == 0:
        return f"{p['nombre']} cuesta {precio} Bs, pero por ahora está agotado 😕."
    return f"{p['nombre']} cuesta {precio} Bs 😊"


def a_stock(p: dict) -> str:
    s = int(p["stock"])
    if s == 0:
        return f"Por ahora {p['nombre']} está agotado 😕."
    return f"De {p['nombre']} tenemos {_unidades(s)} 😊"


def a_specs(p: dict) -> str:
    specs = str(p.get("specs") or "").strip()
    if not specs:
        return f"No tengo detalles técnicos de {p['nombre']} en el catálogo actual."
    return f"{p['nombre']} tiene: {specs}."


ANSWERS = {"ficha": a_ficha, "precio": a_precio, "stock": a_stock, "specs": a_specs}


def _disponibles(items: list[dict], tope: float | None = None) -> list[dict]:
    """Productos con stock (y bajo el tope de precio), del más barato al más caro, máx. 3."""
    ok = [r for r in items if int(r["stock"]) > 0 and (tope is None or r["precio"] <= tope)]
    return sorted(ok, key=lambda r: r["precio"])[:3]


def _lista(items: list[dict]) -> str:
    lines = [f"- {m['nombre']}: {format_precio(m['precio'])} Bs ({int(m['stock'])} en stock)" for m in items]
    return "Esto es lo que tenemos disponible 😊\n" + "\n".join(lines)


# =====================================================================
# 4. VERSIONES DEL CATÁLOGO (anti-memorización)
# =====================================================================
def make_variant(df: pd.DataFrame, rng: random.Random, keep_original: bool) -> list[dict]:
    """Copia del catálogo con precio y stock alterados al azar."""
    recs = df.to_dict("records")
    if keep_original:
        return recs
    out = []
    for r in recs:
        r = dict(r)
        r["precio"] = max(10, int(round(r["precio"] * rng.uniform(0.7, 1.3) / 10) * 10))
        r["stock"] = 0 if rng.random() < 0.12 else rng.randint(1, 25)
        out.append(r)
    return out


# =====================================================================
# 5. GENERADORES DE EJEMPLOS
#    Cada uno arma: contexto (lo que devolvería el RAG) + pregunta + respuesta.
# =====================================================================
def _pack(tienda, ctx, pregunta, respuesta, **meta) -> dict:
    return {"messages": build_messages(tienda, ctx, pregunta, respuesta), "meta": meta}


def _categorias(records):
    return sorted({r["categoria"] for r in records})


def ex_producto(records, tienda, rng, banco) -> dict:
    p = rng.choice(records)
    intent = rng.choices(["ficha", "precio", "stock", "specs"], [4, 2, 2, 2])[0]
    pregunta = _noisy(rng.choice(banco[intent]).format(n=_mention(p, records, rng)), rng)
    same = [r for r in records if r["categoria"] == p["categoria"] and r["id"] != p["id"]]
    other = [r for r in records if r["categoria"] != p["categoria"]]
    ctx = [p] + rng.sample(same, min(len(same), rng.randint(1, 2))) + rng.sample(other, 1)
    rng.shuffle(ctx)
    return _pack(tienda, ctx, pregunta, ANSWERS[intent](p), intent=intent, gold_id=p["id"])


def ex_cat_budget(records, tienda, rng, banco) -> dict:
    cat = rng.choice(_categorias(records))
    items = [r for r in records if r["categoria"] == cat]
    if rng.random() < 0.12:   # presupuesto imposible: nada alcanza
        b = int(round(min(r["precio"] for r in items) * 0.8 / 50) * 50) or 50
    else:
        b = int(round(rng.choice(items)["precio"] * rng.uniform(0.9, 1.5) / 50) * 50) or 50
    within = [r for r in items if r["precio"] <= b]
    beyond = sorted([r for r in items if r["precio"] > b], key=lambda r: r["precio"])
    # el RAG real trae los que cumplen + algunos que se pasan un poco: el modelo debe filtrar
    ctx = rng.sample(within, min(3, len(within))) + beyond[:2] if within else beyond[:4]
    rng.shuffle(ctx)
    pregunta = _noisy(rng.choice(banco["cat_budget"]).format(c=_cat_word(cat, rng), b=_budget_text(b, rng)), rng)
    ok = _disponibles(ctx, b)
    if ok:
        resp = _lista(ok)
    else:
        resp = f"No encuentro {cat.lower()} disponibles por menos de {format_precio(b)} Bs en el catálogo actual."
        barato = _disponibles(ctx)
        if barato:
            resp += f" La opción más económica es {barato[0]['nombre']} a {format_precio(barato[0]['precio'])} Bs."
    return _pack(tienda, ctx, pregunta, resp, intent="cat_budget", gold_id=None, presupuesto=b, n_ok=len(ok))


def ex_cat_only(records, tienda, rng, banco) -> dict:
    cat = rng.choice(_categorias(records))
    items = [r for r in records if r["categoria"] == cat]
    ctx = rng.sample(items, min(5, len(items)))
    pregunta = _noisy(rng.choice(banco["cat_only"]).format(c=_cat_word(cat, rng)), rng)
    ok = _disponibles(ctx)
    resp = _lista(ok) if ok else f"Por ahora no tenemos {cat.lower()} disponibles 😕."
    return _pack(tienda, ctx, pregunta, resp, intent="cat_only", gold_id=None)


def ex_cat_cheapest(records, tienda, rng, banco) -> dict:
    cat = rng.choice(_categorias(records))
    items = sorted([r for r in records if r["categoria"] == cat], key=lambda r: r["precio"])
    ctx = items[:4]
    rng.shuffle(ctx)
    pregunta = _noisy(rng.choice(banco["cat_cheapest"]).format(c=_cat_word(cat, rng)), rng)
    ok = _disponibles(ctx)[:1]
    if ok:
        p = ok[0]
        resp = (f"La opción más económica que tenemos es {p['nombre']}: "
                f"{format_precio(p['precio'])} Bs, con {_unidades(int(p['stock']))}.")
    else:
        resp = f"Por ahora no tenemos {cat.lower()} disponibles 😕."
    return _pack(tienda, ctx, pregunta, resp, intent="cat_cheapest", gold_id=None)


def ex_cat_attr(records, tienda, rng, banco) -> dict:
    cat = rng.choice(_categorias(records))
    items = [r for r in records if r["categoria"] == cat and str(r["specs"]).strip()]
    for _ in range(5):   # busca un atributo que distinga unos productos de otros
        anchor = rng.choice(items)
        original = rng.choice([t.strip() for t in anchor["specs"].split(",") if t.strip()])
        matches = [r for r in items if original.lower() in _tokens(r["specs"])]
        if len(matches) < len(items):
            break
    others = [r for r in items if r not in matches]
    ctx = rng.sample(matches, min(3, len(matches))) + rng.sample(others, min(2, len(others)))
    rng.shuffle(ctx)
    pregunta = _noisy(rng.choice(banco["cat_attr"]).format(c=_cat_word(cat, rng), a=original), rng)
    ok = _disponibles([r for r in ctx if original.lower() in _tokens(r["specs"])])
    resp = _lista(ok) if ok else f"Por ahora no tenemos {cat.lower()} con {original} disponibles 😕."
    return _pack(tienda, ctx, pregunta, resp, intent="cat_attr", gold_id=None, atributo=original)


def ex_marca(records, tienda, rng, banco) -> dict:
    marca = rng.choice(sorted({r["marca"] for r in records if r["marca"]}))
    items = [r for r in records if r["marca"] == marca]
    ctx = rng.sample(items, min(5, len(items)))
    pregunta = _noisy(rng.choice(banco["marca"]).format(m=marca), rng)
    ok = _disponibles(ctx)
    resp = _lista(ok) if ok else f"Por ahora no tenemos productos {marca} disponibles 😕."
    return _pack(tienda, ctx, pregunta, resp, intent="marca", gold_id=None)


# =====================================================================
# 6. MEZCLA DE LA PARTE 2A
#    Por cada producto del catálogo: 1 ejemplo de producto + 2 de descubrimiento
#    (≈ 2/3 descubrimiento, 1/3 producto). En 2B se suman los casos de honestidad,
#    y la mezcla final queda ≈ 50% descubrimiento / 25% producto / 25% honestidad.
# =====================================================================
DESCUBRIMIENTO = {
    "cat_budget": (ex_cat_budget, 40),
    "cat_attr": (ex_cat_attr, 20),
    "cat_only": (ex_cat_only, 15),
    "cat_cheapest": (ex_cat_cheapest, 15),
    "marca": (ex_marca, 10),
}


def ejemplos_2a(df: pd.DataFrame, tienda: str, rng: random.Random,
                keep_original: bool = False, banco: dict | None = None) -> list[dict]:
    """Ejemplos de UNA versión del catálogo. banco=Q_TRAIN (default) o Q_TEST."""
    banco = banco or Q_TRAIN
    records = make_variant(df, rng, keep_original)

    gens = dict(DESCUBRIMIENTO)
    if not any(r["marca"] for r in records):
        gens.pop("marca")
    if not any(str(r["specs"]).strip() for r in records):
        gens.pop("cat_attr")
    funcs, weights = zip(*gens.values())

    out = [ex_producto(records, tienda, rng, banco) for _ in records]
    out += [rng.choices(funcs, weights)[0](records, tienda, rng, banco) for _ in range(2 * len(records))]
    rng.shuffle(out)
    return out