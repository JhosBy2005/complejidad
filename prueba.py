# -*- coding: utf-8 -*-
"""
Aplicacion de consola para deteccion de transacciones bancarias sospechosas.

El usuario ingresa datos financieros en CSV o por consola. El sistema construye
grafos internos y aplica algoritmos para devolver alertas de fraude.

Algoritmos usados:
1. Union-Find / UFDS: uso de dispositivo compartido.
2. BFS: posible tarjeta comprometida.
3. Tarjan SCC: identidad sintetica o cuentas mula.
"""

from __future__ import annotations

import csv
import html
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple


REQUIRED_COLUMNS = [
    "id_transaccion",
    "cuenta",
    "tarjeta",
    "comercio",
    "dispositivo",
    "atributo_identidad",
    "tiempo_minutos",
    "monto",
]

IEEE_TRANSACTION_FILES = ["train_transaction.csv", "train_transaction_sample.csv"]
IEEE_IDENTITY_FILES = ["train_identity.csv", "train_identity_sample.csv"]


@dataclass
class Transaccion:
    id_transaccion: str
    cuenta: str
    tarjeta: str
    comercio: str
    dispositivo: str
    atributo_identidad: str
    tiempo_minutos: int
    monto: float
    banco: str = "BANCO_PERU"
    ciudad: str = "LIMA"


@dataclass
class Alerta:
    codigo: str
    tipo: str
    nivel: str
    algoritmo: str
    descripcion: str
    elementos: List[str]
    complejidad: str


class UnionFind:
    """Union-Find con union por rango y compresion de caminos."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def grupos(self) -> Dict[str, List[str]]:
        resultado: DefaultDict[str, List[str]] = defaultdict(list)
        for item in list(self.parent):
            resultado[self.find(item)].append(item)
        return dict(resultado)


class DetectorFraudeConsola:
    def __init__(self) -> None:
        self.transacciones: List[Transaccion] = []
        self.alertas: List[Alerta] = []
        self.grafo_dispositivo: DefaultDict[str, Set[str]] = defaultdict(set)
        self.grafo_transacciones: DefaultDict[str, Set[str]] = defaultdict(set)
        self.grafo_identidad: DefaultDict[str, Set[str]] = defaultdict(set)
        self.por_tarjeta: DefaultDict[str, List[Transaccion]] = defaultdict(list)
        self.por_dispositivo: DefaultDict[str, Set[str]] = defaultdict(set)
        self.por_identidad: DefaultDict[str, Set[str]] = defaultdict(set)

    def cargar_csv(self, ruta: str) -> None:
        path = Path(ruta).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {ruta}")
        if path.is_dir():
            raise IsADirectoryError("Ingrese la ruta de un archivo CSV, no una carpeta.")

        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("El CSV no tiene cabecera.")

            columnas = set(reader.fieldnames)
            if es_csv_ieee_transacciones(columnas):
                ruta_identidad = buscar_archivo_opcional(path.parent, IEEE_IDENTITY_FILES)
                self.cargar_ieee_csv(str(path), str(ruta_identidad) if ruta_identidad else None)
                return
            if es_csv_ieee_identidad(columnas):
                ruta_transacciones = resolver_archivo(path.parent, IEEE_TRANSACTION_FILES)
                self.cargar_ieee_csv(str(ruta_transacciones), str(path))
                return

            faltantes = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
            if faltantes:
                raise ValueError(f"Faltan columnas obligatorias: {faltantes}")
            self.transacciones = [self._row_a_transaccion(row) for row in reader]

        self.alertas = []
        self.construir_grafos()
        print(f"Datos cargados: {len(self.transacciones)} transacciones.")

    def cargar_dataset_ieee(self, carpeta_dataset: str = "", limite: Optional[int] = None) -> None:
        carpeta = Path(carpeta_dataset).expanduser() if carpeta_dataset else carpeta_dataset_por_defecto()
        if not carpeta.exists():
            raise FileNotFoundError(f"No existe la carpeta Dataset: {carpeta}")
        ruta_transacciones = resolver_archivo(carpeta, IEEE_TRANSACTION_FILES)
        ruta_identidad = resolver_archivo(carpeta, IEEE_IDENTITY_FILES)
        self.cargar_ieee_csv(str(ruta_transacciones), str(ruta_identidad), limite=limite)

    def cargar_ieee_csv(
        self,
        ruta_transacciones: str,
        ruta_identidad: Optional[str] = None,
        limite: Optional[int] = None,
    ) -> None:
        path_tx = Path(ruta_transacciones).expanduser()
        if not path_tx.exists():
            raise FileNotFoundError(f"No existe el archivo de transacciones: {ruta_transacciones}")

        identidades: Dict[str, Dict[str, str]] = {}
        if ruta_identidad:
            path_id = Path(ruta_identidad).expanduser()
            if not path_id.exists():
                raise FileNotFoundError(f"No existe el archivo de identidad: {ruta_identidad}")
            with path_id.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    identidades[row.get("TransactionID", "")] = row

        with path_tx.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("El CSV de transacciones no tiene cabecera.")
            necesarias = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card1"]
            faltantes = [c for c in necesarias if c not in reader.fieldnames]
            if faltantes:
                raise ValueError(f"El CSV IEEE no tiene columnas necesarias: {faltantes}")

            self.transacciones = []
            for i, row in enumerate(reader):
                if limite is not None and i >= limite:
                    break
                identidad = identidades.get(row.get("TransactionID", ""), {})
                self.transacciones.append(self._ieee_row_a_transaccion(row, identidad))

        self.alertas = []
        self.construir_grafos()
        print(f"Dataset IEEE cargado: {len(self.transacciones)} transacciones.")
        print(f"Archivo transacciones: {path_tx}")
        if ruta_identidad:
            print(f"Archivo identidad    : {ruta_identidad}")

    def cargar_simulacion(self) -> None:
        self.transacciones = datos_simulados()
        self.alertas = []
        self.construir_grafos()
        print(f"Simulacion cargada: {len(self.transacciones)} transacciones.")

    def ingresar_manual(self) -> None:
        print("\nIngreso manual de transaccion")
        row = {}
        for col in REQUIRED_COLUMNS:
            row[col] = input(f"{col}: ").strip()
        row["banco"] = input("banco (opcional): ").strip() or "BANCO_PERU"
        row["ciudad"] = input("ciudad (opcional): ").strip() or "LIMA"
        self.transacciones.append(self._row_a_transaccion(row))
        self.construir_grafos()
        print("Transaccion agregada.")

    def _row_a_transaccion(self, row: Dict[str, str]) -> Transaccion:
        return Transaccion(
            id_transaccion=row["id_transaccion"].strip(),
            cuenta=row["cuenta"].strip(),
            tarjeta=row["tarjeta"].strip(),
            comercio=row["comercio"].strip(),
            dispositivo=row["dispositivo"].strip(),
            atributo_identidad=row["atributo_identidad"].strip(),
            tiempo_minutos=int(float(row["tiempo_minutos"])),
            monto=float(row["monto"]),
            banco=row.get("banco", "BANCO_PERU").strip() or "BANCO_PERU",
            ciudad=row.get("ciudad", "LIMA").strip() or "LIMA",
        )

    def _ieee_row_a_transaccion(self, row: Dict[str, str], identidad: Dict[str, str]) -> Transaccion:
        transaction_id = limpiar(row.get("TransactionID"))
        card1 = limpiar(row.get("card1"))
        card2 = limpiar(row.get("card2"))
        card4 = limpiar(row.get("card4"))
        card6 = limpiar(row.get("card6"))
        addr1 = limpiar(row.get("addr1"))
        addr2 = limpiar(row.get("addr2"))
        email = limpiar(row.get("P_emaildomain") or row.get("R_emaildomain"))
        product = limpiar(row.get("ProductCD"))
        device_type = limpiar(identidad.get("DeviceType"))
        device_info = limpiar(identidad.get("DeviceInfo"))
        browser = limpiar(identidad.get("id_31"))
        os_name = limpiar(identidad.get("id_30"))
        resolution = limpiar(identidad.get("id_33"))
        identity_region = limpiar(identidad.get("id_13"))

        cuenta = f"ACC-{card1}-{addr1}-{email}"
        tarjeta = f"CARD-{card1}-{card2}-{card4}-{card6}"
        comercio = f"COM-{product}-{addr1}-{addr2}"

        if device_info != "UNK" or device_type != "UNK":
            dispositivo = compactar_id("DEV", [device_type, device_info, browser, resolution])
        else:
            dispositivo = compactar_id("DEV", ["SIN_IDENTIDAD", card1, card2, addr1])

        if os_name != "UNK" or browser != "UNK" or resolution != "UNK":
            atributo_identidad = compactar_id("IDENT", [os_name, browser, resolution, device_type, identity_region])
        else:
            atributo_identidad = compactar_id("IDENT", [email, addr1, addr2, card4, card1])

        return Transaccion(
            id_transaccion=f"TX-{transaction_id}",
            cuenta=cuenta,
            tarjeta=tarjeta,
            comercio=comercio,
            dispositivo=dispositivo,
            atributo_identidad=atributo_identidad,
            tiempo_minutos=int(float(row.get("TransactionDT") or 0) // 60),
            monto=float(row.get("TransactionAmt") or 0),
            banco=banco_por_tarjeta(card1),
            ciudad=ciudad_por_direccion(addr1),
        )

    def construir_grafos(self) -> None:
        self.grafo_dispositivo.clear()
        self.grafo_transacciones.clear()
        self.grafo_identidad.clear()
        self.por_tarjeta.clear()
        self.por_dispositivo.clear()
        self.por_identidad.clear()

        for t in self.transacciones:
            self.grafo_dispositivo[t.dispositivo].add(t.cuenta)
            self.grafo_dispositivo[t.cuenta].add(t.dispositivo)
            self.por_dispositivo[t.dispositivo].add(t.cuenta)

            self.grafo_transacciones[t.tarjeta].add(t.id_transaccion)
            self.grafo_transacciones[t.id_transaccion].add(t.comercio)
            self.grafo_transacciones[t.id_transaccion].add(t.cuenta)
            self.por_tarjeta[t.tarjeta].append(t)

            self.por_identidad[t.atributo_identidad].add(t.cuenta)

        for _atributo, cuentas in self.por_identidad.items():
            cuentas_ordenadas = sorted(cuentas)
            for i in range(len(cuentas_ordenadas)):
                for j in range(i + 1, len(cuentas_ordenadas)):
                    a = cuentas_ordenadas[i]
                    b = cuentas_ordenadas[j]
                    self.grafo_identidad[a].add(b)
                    self.grafo_identidad[b].add(a)

    def detectar_fraude_dispositivo(self, minimo_cuentas: int = 3) -> List[Alerta]:
        uf = UnionFind()
        for t in self.transacciones:
            uf.add(t.cuenta)

        for _dispositivo, cuentas in self.por_dispositivo.items():
            cuentas_lista = sorted(cuentas)
            if len(cuentas_lista) >= 2:
                base = cuentas_lista[0]
                for cuenta in cuentas_lista[1:]:
                    uf.union(base, cuenta)

        alertas: List[Alerta] = []
        for _root, cuentas in uf.grupos().items():
            if len(cuentas) >= minimo_cuentas:
                dispositivos = sorted({t.dispositivo for t in self.transacciones if t.cuenta in cuentas})
                nivel = "Alto" if len(cuentas) >= 5 else "Medio"
                alertas.append(Alerta(
                    codigo="",
                    tipo="Uso de dispositivo compartido",
                    nivel=nivel,
                    algoritmo="Union-Find / UFDS",
                    descripcion=f"{len(cuentas)} cuentas aparecen conectadas por uno o mas dispositivos comunes.",
                    elementos=cuentas + dispositivos,
                    complejidad="O((V+E) alpha(n))",
                ))
        return alertas

    def detectar_tarjeta_comprometida(
        self,
        ventana_minutos: int = 30,
        minimo_transacciones: int = 3,
        minimo_comercios: int = 3,
    ) -> List[Alerta]:
        alertas: List[Alerta] = []
        for tarjeta, transacciones in self.por_tarjeta.items():
            ordenadas = sorted(transacciones, key=lambda x: x.tiempo_minutos)
            for inicio in range(len(ordenadas)):
                ventana = []
                for t in ordenadas[inicio:]:
                    if t.tiempo_minutos - ordenadas[inicio].tiempo_minutos <= ventana_minutos:
                        ventana.append(t)
                    else:
                        break
                comercios = {t.comercio for t in ventana}
                if len(ventana) >= minimo_transacciones and len(comercios) >= minimo_comercios:
                    alcanzados = self.bfs(tarjeta, profundidad_maxima=2)
                    elementos = [tarjeta] + [t.id_transaccion for t in ventana] + sorted(comercios)
                    nivel = "Alto" if len(ventana) >= 4 else "Medio"
                    alertas.append(Alerta(
                        codigo="",
                        tipo="Posible tarjeta comprometida",
                        nivel=nivel,
                        algoritmo="BFS",
                        descripcion=(
                            f"La tarjeta {tarjeta} alcanza {len(comercios)} comercios "
                            f"y {len(ventana)} transacciones dentro de {ventana_minutos} minutos. "
                            f"Nodos alcanzados por BFS: {len(alcanzados)}."
                        ),
                        elementos=elementos,
                        complejidad="O(V+E)",
                    ))
                    break
        return alertas

    def bfs(self, inicio: str, profundidad_maxima: int = 2) -> Set[str]:
        visitados = {inicio}
        cola = deque([(inicio, 0)])
        while cola:
            nodo, profundidad = cola.popleft()
            if profundidad == profundidad_maxima:
                continue
            for vecino in self.grafo_transacciones.get(nodo, set()):
                if vecino not in visitados:
                    visitados.add(vecino)
                    cola.append((vecino, profundidad + 1))
        return visitados

    def detectar_identidad_sintetica(self, minimo_cuentas: int = 3) -> List[Alerta]:
        componentes = self.tarjan_scc(self.grafo_identidad)
        alertas: List[Alerta] = []
        for componente in componentes:
            if len(componente) >= minimo_cuentas:
                nivel = "Alto" if len(componente) >= 5 else "Medio"
                alertas.append(Alerta(
                    codigo="",
                    tipo="Posible identidad sintetica o cuentas mula",
                    nivel=nivel,
                    algoritmo="Tarjan SCC",
                    descripcion=(
                        f"Se encontro una componente fuertemente conexa de "
                        f"{len(componente)} cuentas con atributos compartidos."
                    ),
                    elementos=sorted(componente),
                    complejidad="O(V+E)",
                ))
        return alertas

    def tarjan_scc(self, grafo: Dict[str, Set[str]]) -> List[Set[str]]:
        indice = 0
        pila: List[str] = []
        en_pila: Set[str] = set()
        indices: Dict[str, int] = {}
        bajos: Dict[str, int] = {}
        componentes: List[Set[str]] = []

        def conectar(v: str) -> None:
            nonlocal indice
            indices[v] = indice
            bajos[v] = indice
            indice += 1
            pila.append(v)
            en_pila.add(v)
            for w in grafo.get(v, set()):
                if w not in indices:
                    conectar(w)
                    bajos[v] = min(bajos[v], bajos[w])
                elif w in en_pila:
                    bajos[v] = min(bajos[v], indices[w])
            if bajos[v] == indices[v]:
                componente = set()
                while True:
                    w = pila.pop()
                    en_pila.remove(w)
                    componente.add(w)
                    if w == v:
                        break
                componentes.append(componente)

        for nodo in grafo:
            if nodo not in indices:
                conectar(nodo)
        return componentes

    def ejecutar_detecciones(self) -> List[Alerta]:
        alertas = []
        alertas.extend(self.detectar_fraude_dispositivo())
        alertas.extend(self.detectar_tarjeta_comprometida())
        alertas.extend(self.detectar_identidad_sintetica())
        for i, alerta in enumerate(alertas, 1):
            alerta.codigo = f"ALERTA {i:03d}"
        self.alertas = alertas
        return alertas

    def resumen(self) -> None:
        cuentas = {t.cuenta for t in self.transacciones}
        tarjetas = {t.tarjeta for t in self.transacciones}
        comercios = {t.comercio for t in self.transacciones}
        dispositivos = {t.dispositivo for t in self.transacciones}
        print("\nRESUMEN DE DATOS")
        print("-" * 60)
        print(f"Transacciones : {len(self.transacciones)}")
        print(f"Cuentas       : {len(cuentas)}")
        print(f"Tarjetas      : {len(tarjetas)}")
        print(f"Comercios     : {len(comercios)}")
        print(f"Dispositivos  : {len(dispositivos)}")

    def imprimir_alertas(self) -> None:
        if not self.alertas:
            print("No hay alertas. Ejecute primero las detecciones.")
            return
        print("\nALERTAS DE FRAUDE")
        print("-" * 60)
        for alerta in self.alertas:
            print(alerta.codigo)
            print(f"Tipo       : {alerta.tipo}")
            print(f"Nivel      : {alerta.nivel}")
            print(f"Algoritmo  : {alerta.algoritmo}")
            print(f"Motivo     : {alerta.descripcion}")
            print(f"Elementos  : {', '.join(alerta.elementos[:12])}")
            print(f"Complejidad: {alerta.complejidad}")
            print("-" * 60)

    def exportar_reporte(self, ruta: str = "reporte_alertas.csv") -> None:
        if not self.alertas:
            print("No hay alertas para exportar.")
            return
        with open(ruta, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["codigo", "tipo", "nivel", "algoritmo", "descripcion", "elementos", "complejidad"])
            for a in self.alertas:
                writer.writerow([
                    a.codigo,
                    a.tipo,
                    a.nivel,
                    a.algoritmo,
                    a.descripcion,
                    " | ".join(a.elementos),
                    a.complejidad,
                ])
        print(f"Reporte exportado en: {ruta}")

    def generar_grafos(self, carpeta_salida: str = "Grafos", limite_nodos: int = 45) -> List[Path]:
        if not self.transacciones:
            raise ValueError("Primero cargue datos antes de generar grafos.")
        salida = Path(carpeta_salida)
        if not salida.is_absolute():
            salida = Path(__file__).resolve().parent.parent / salida
        salida.mkdir(parents=True, exist_ok=True)

        archivos: List[Path] = []
        nodos, aristas, descripcion = self._subgrafo_dispositivo(limite_nodos)
        ruta = salida / "grafo_1_dispositivo_compartido.svg"
        guardar_svg_grafo(ruta, "Caso 1: uso de dispositivo compartido", descripcion, nodos, aristas)
        archivos.append(ruta)

        nodos, aristas, descripcion = self._subgrafo_tarjeta(limite_nodos)
        ruta = salida / "grafo_2_tarjeta_comprometida.svg"
        guardar_svg_grafo(ruta, "Caso 2: posible tarjeta comprometida", descripcion, nodos, aristas)
        archivos.append(ruta)

        nodos, aristas, descripcion = self._subgrafo_identidad(limite_nodos)
        ruta = salida / "grafo_3_identidad_sintetica.svg"
        guardar_svg_grafo(ruta, "Caso 3: identidad sintetica o cuentas mula", descripcion, nodos, aristas)
        archivos.append(ruta)

        print("Grafos generados:")
        for archivo in archivos:
            print(f"- {archivo}")
        return archivos

    def _subgrafo_dispositivo(self, limite_nodos: int) -> Tuple[Dict[str, str], List[Tuple[str, str]], str]:
        dispositivo, cuentas = max(self.por_dispositivo.items(), key=lambda item: len(item[1]))
        cuentas_ordenadas = sorted(cuentas)[: max(1, limite_nodos - 1)]
        nodos = {dispositivo: "Dispositivo"}
        aristas: List[Tuple[str, str]] = []
        for cuenta in cuentas_ordenadas:
            nodos[cuenta] = "Cuenta"
            aristas.append((dispositivo, cuenta))
        descripcion = "Subgrafo elegido: dispositivo con mayor numero de cuentas asociadas. Explica Union-Find / UFDS."
        return nodos, aristas, descripcion

    def _subgrafo_tarjeta(
        self,
        limite_nodos: int,
        ventana_minutos: int = 30,
    ) -> Tuple[Dict[str, str], List[Tuple[str, str]], str]:
        mejor_tarjeta = ""
        mejor_ventana: List[Transaccion] = []
        mejor_puntaje = (-1, -1)
        for tarjeta, transacciones in self.por_tarjeta.items():
            ordenadas = sorted(transacciones, key=lambda x: x.tiempo_minutos)
            for inicio in range(len(ordenadas)):
                ventana = []
                for t in ordenadas[inicio:]:
                    if t.tiempo_minutos - ordenadas[inicio].tiempo_minutos <= ventana_minutos:
                        ventana.append(t)
                    else:
                        break
                puntaje = (len({t.comercio for t in ventana}), len(ventana))
                if puntaje > mejor_puntaje:
                    mejor_tarjeta = tarjeta
                    mejor_ventana = ventana
                    mejor_puntaje = puntaje

        if not mejor_ventana:
            mejor_tarjeta, mejor_ventana = max(self.por_tarjeta.items(), key=lambda item: len(item[1]))

        nodos = {mejor_tarjeta: "Tarjeta"}
        aristas: List[Tuple[str, str]] = []
        for t in mejor_ventana:
            if len(nodos) >= limite_nodos:
                break
            nodos[t.id_transaccion] = "Transaccion"
            nodos[t.comercio] = "Comercio"
            nodos[t.cuenta] = "Cuenta"
            aristas.append((mejor_tarjeta, t.id_transaccion))
            aristas.append((t.id_transaccion, t.comercio))
            aristas.append((t.id_transaccion, t.cuenta))
        descripcion = f"Subgrafo elegido: tarjeta con mayor movimiento hacia comercios dentro de {ventana_minutos} minutos. Explica BFS."
        return nodos, aristas, descripcion

    def _subgrafo_identidad(self, limite_nodos: int) -> Tuple[Dict[str, str], List[Tuple[str, str]], str]:
        atributo, cuentas = max(self.por_identidad.items(), key=lambda item: len(item[1]))
        cuentas_ordenadas = sorted(cuentas)[: max(1, limite_nodos - 1)]
        nodos = {atributo: "Identidad"}
        aristas: List[Tuple[str, str]] = []
        for cuenta in cuentas_ordenadas:
            nodos[cuenta] = "Cuenta"
            aristas.append((atributo, cuenta))
        for i in range(len(cuentas_ordenadas) - 1):
            aristas.append((cuentas_ordenadas[i], cuentas_ordenadas[i + 1]))
        descripcion = "Subgrafo elegido: atributo de identidad compartido por mas cuentas. Explica Tarjan SCC."
        return nodos, aristas, descripcion


def carpeta_dataset_por_defecto() -> Path:
    script = Path(__file__).resolve()
    candidatos = [
        script.parents[2] / "Dataset",
        script.parents[1] / "Dataset",
        Path.cwd() / "Dataset",
    ]
    for candidato in candidatos:
        if candidato.exists():
            return candidato
    return candidatos[0]


def resolver_archivo(carpeta: Path, nombres: List[str]) -> Path:
    for nombre in nombres:
        candidato = carpeta / nombre
        if candidato.exists():
            return candidato
    raise FileNotFoundError(f"No se encontro ninguno de estos archivos en {carpeta}: {', '.join(nombres)}")


def buscar_archivo_opcional(carpeta: Path, nombres: List[str]) -> Optional[Path]:
    for nombre in nombres:
        candidato = carpeta / nombre
        if candidato.exists():
            return candidato
    return None


def es_csv_ieee_transacciones(columnas: Set[str]) -> bool:
    necesarias = {"TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card1"}
    return necesarias.issubset(columnas)


def es_csv_ieee_identidad(columnas: Set[str]) -> bool:
    necesarias = {"TransactionID", "DeviceType", "DeviceInfo"}
    return necesarias.issubset(columnas) and "TransactionAmt" not in columnas


def limpiar(valor: Optional[str]) -> str:
    texto = str(valor or "").strip()
    if not texto or texto.lower() in {"nan", "none", "null"}:
        return "UNK"
    if texto.endswith(".0"):
        texto = texto[:-2]
    return texto.replace(" ", "_")


def compactar_id(prefijo: str, partes: Iterable[str]) -> str:
    utiles = [limpiar(p) for p in partes if limpiar(p) != "UNK"]
    if not utiles:
        utiles = ["UNK"]
    return f"{prefijo}-{'-'.join(utiles[:5])}"


def banco_por_tarjeta(card1: str) -> str:
    bancos = ["BCP", "BBVA", "INTERBANK", "SCOTIABANK", "BANCO_PICHINCHA"]
    try:
        return bancos[int(float(card1)) % len(bancos)]
    except ValueError:
        return "BANCO_PERU"


def ciudad_por_direccion(addr1: str) -> str:
    ciudades = ["LIMA", "CALLAO", "AREQUIPA", "TRUJILLO", "CUSCO", "PIURA"]
    try:
        return ciudades[int(float(addr1)) % len(ciudades)]
    except ValueError:
        return "LIMA"


def guardar_svg_grafo(
    ruta: Path,
    titulo: str,
    descripcion: str,
    nodos: Dict[str, str],
    aristas: List[Tuple[str, str]],
) -> None:
    posiciones, ancho, alto = calcular_layout(nodos)
    colores = {
        "Cuenta": ("#E8F3FF", "#1D5D9B"),
        "Tarjeta": ("#FFF1D6", "#9A5B00"),
        "Transaccion": ("#EAF8E6", "#3D7A2B"),
        "Comercio": ("#F3E8FF", "#6D3CA5"),
        "Dispositivo": ("#FFE7E7", "#A63A3A"),
        "Identidad": ("#E9F7F3", "#27715E"),
    }
    lineas = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{ancho}" height="{alto}" viewBox="0 0 {ancho} {alto}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#667085" />',
        "</marker>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#FAFBFC" />',
        f'<text x="40" y="42" font-family="Arial" font-size="24" font-weight="700" fill="#1F2937">{esc(titulo)}</text>',
        f'<text x="40" y="72" font-family="Arial" font-size="14" fill="#4B5563">{esc(descripcion)}</text>',
    ]
    for origen, destino in aristas:
        if origen not in posiciones or destino not in posiciones:
            continue
        x1, y1 = posiciones[origen]
        x2, y2 = posiciones[destino]
        lineas.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#98A2B3" stroke-width="1.6" marker-end="url(#arrow)" />'
        )
    for nodo, tipo in nodos.items():
        x, y = posiciones[nodo]
        relleno, borde = colores.get(tipo, ("#FFFFFF", "#475467"))
        w = 210
        h = 48
        lineas.extend([
            f'<g><title>{esc(nodo)}</title>',
            f'<rect x="{x - w / 2}" y="{y - h / 2}" width="{w}" height="{h}" rx="8" fill="{relleno}" stroke="{borde}" stroke-width="1.4" />',
            f'<text x="{x}" y="{y - 4}" text-anchor="middle" font-family="Arial" font-size="12" font-weight="700" fill="{borde}">{esc(tipo)}</text>',
            f'<text x="{x}" y="{y + 14}" text-anchor="middle" font-family="Arial" font-size="11" fill="#344054">{esc(etiqueta_corta(nodo))}</text>',
            "</g>",
        ])
    lineas.append("</svg>")
    ruta.write_text("\n".join(lineas), encoding="utf-8")


def calcular_layout(nodos: Dict[str, str]) -> Tuple[Dict[str, Tuple[float, float]], int, int]:
    if any(tipo == "Tarjeta" for tipo in nodos.values()):
        orden = ["Tarjeta", "Transaccion", "Comercio", "Cuenta"]
    elif any(tipo == "Dispositivo" for tipo in nodos.values()):
        orden = ["Dispositivo", "Cuenta"]
    elif any(tipo == "Identidad" for tipo in nodos.values()):
        orden = ["Identidad", "Cuenta"]
    else:
        orden = sorted(set(nodos.values()))

    grupos: DefaultDict[str, List[str]] = defaultdict(list)
    for nodo, tipo in nodos.items():
        grupos[tipo].append(nodo)
    for tipo in grupos:
        grupos[tipo].sort()

    columnas = [tipo for tipo in orden if grupos.get(tipo)]
    max_filas = max((len(grupos[tipo]) for tipo in columnas), default=1)
    ancho = max(1000, 260 * max(1, len(columnas)) + 120)
    alto = max(650, 125 + max_filas * 72)
    posiciones: Dict[str, Tuple[float, float]] = {}
    margen_x = 140
    espacio_x = (ancho - margen_x * 2) / max(1, len(columnas) - 1)
    for col, tipo in enumerate(columnas):
        x = ancho / 2 if len(columnas) == 1 else margen_x + col * espacio_x
        elementos = grupos[tipo]
        espacio_y = (alto - 180) / max(1, len(elementos) + 1)
        for fila, nodo in enumerate(elementos, 1):
            y = 110 + fila * espacio_y
            posiciones[nodo] = (x, y)
    return posiciones, ancho, alto


def etiqueta_corta(texto: str, max_len: int = 28) -> str:
    if len(texto) <= max_len:
        return texto
    return texto[: max_len - 3] + "..."


def esc(texto: str) -> str:
    return html.escape(str(texto), quote=True)


def datos_simulados() -> List[Transaccion]:
    return [
        Transaccion("T001", "C001", "CARD-01", "M001", "DEV-001", "ID-A", 10, 120.0, "BCP", "LIMA"),
        Transaccion("T002", "C002", "CARD-02", "M002", "DEV-001", "ID-B", 12, 80.0, "BBVA", "LIMA"),
        Transaccion("T003", "C003", "CARD-03", "M003", "DEV-001", "ID-C", 14, 95.0, "INTERBANK", "LIMA"),
        Transaccion("T004", "C004", "CARD-04", "M004", "DEV-001", "ID-D", 16, 60.0, "SCOTIABANK", "LIMA"),
        Transaccion("T005", "C010", "CARD-X", "M010", "DEV-010", "ID-X1", 100, 200.0, "BCP", "LIMA"),
        Transaccion("T006", "C010", "CARD-X", "M011", "DEV-010", "ID-X1", 108, 350.0, "BCP", "CALLAO"),
        Transaccion("T007", "C010", "CARD-X", "M012", "DEV-010", "ID-X1", 119, 500.0, "BCP", "AREQUIPA"),
        Transaccion("T008", "C020", "CARD-20", "M020", "DEV-020", "EMAIL-COMUN", 210, 40.0, "BBVA", "LIMA"),
        Transaccion("T009", "C021", "CARD-21", "M021", "DEV-021", "EMAIL-COMUN", 215, 70.0, "BBVA", "LIMA"),
        Transaccion("T010", "C022", "CARD-22", "M022", "DEV-022", "EMAIL-COMUN", 220, 65.0, "BBVA", "LIMA"),
        Transaccion("T011", "C023", "CARD-23", "M023", "DEV-023", "EMAIL-COMUN", 225, 90.0, "BBVA", "LIMA"),
        Transaccion("T012", "C030", "CARD-30", "M030", "DEV-030", "ID-N1", 300, 25.0, "PICHINCHA", "LIMA"),
        Transaccion("T013", "C031", "CARD-31", "M031", "DEV-031", "ID-N2", 400, 55.0, "BCP", "CUSCO"),
    ]


def menu() -> None:
    detector = DetectorFraudeConsola()
    while True:
        print(
            """
MENU PRINCIPAL - DETECCION DE FRAUDE BANCARIO
1. Cargar dataset IEEE pegando las 2 rutas CSV
2. Cargar un CSV normal del proyecto
3. Ingresar transaccion manual
4. Ver resumen de datos
5. Ejecutar algoritmos de deteccion
6. Ver alertas
7. Exportar reporte CSV
8. Generar grafos SVG
9. Usar simulacion incluida
0. Salir
"""
        )
        opcion = input("Seleccione una opcion: ").strip()
        try:
            if opcion == "1":
                ruta_transacciones = input("Ruta train_transaction_sample.csv: ").strip().strip('"')
                ruta_identidad = input("Ruta train_identity_sample.csv: ").strip().strip('"')
                limite_txt = input("Limite de filas [todas las disponibles]: ").strip()
                detector.cargar_ieee_csv(
                    ruta_transacciones,
                    ruta_identidad,
                    limite=int(limite_txt) if limite_txt else None,
                )
            elif opcion == "2":
                ruta = input("Ruta del CSV normal: ").strip().strip('"')
                detector.cargar_csv(ruta)
            elif opcion == "3":
                detector.ingresar_manual()
            elif opcion == "4":
                detector.resumen()
            elif opcion == "5":
                detector.ejecutar_detecciones()
                print("Detecciones ejecutadas.")
            elif opcion == "6":
                detector.imprimir_alertas()
            elif opcion == "7":
                ruta = input("Nombre del reporte [reporte_alertas.csv]: ").strip()
                detector.exportar_reporte(ruta or "reporte_alertas.csv")
            elif opcion == "8":
                carpeta = input("Carpeta de salida [Grafos]: ").strip()
                limite_txt = input("Limite de nodos por grafo [45]: ").strip()
                detector.generar_grafos(carpeta or "Grafos", limite_nodos=int(limite_txt) if limite_txt else 45)
            elif opcion == "9":
                detector.cargar_simulacion()
            elif opcion == "0":
                print("Fin del programa.")
                break
            else:
                print("Opcion no valida.")
        except Exception as exc:
            print(f"Error: {exc}")


def demo() -> None:
    detector = DetectorFraudeConsola()
    detector.cargar_simulacion()
    detector.resumen()
    detector.ejecutar_detecciones()
    detector.imprimir_alertas()
    detector.exportar_reporte("reporte_alertas_demo.csv")
    detector.generar_grafos("Grafos")


def demo_ieee() -> None:
    detector = DetectorFraudeConsola()
    detector.cargar_dataset_ieee()
    detector.resumen()
    detector.ejecutar_detecciones()
    detector.imprimir_alertas()
    detector.exportar_reporte("reporte_alertas_ieee.csv")
    detector.generar_grafos("Grafos")


if __name__ == "__main__":
    if "--ieee" in sys.argv or "--real" in sys.argv:
        demo_ieee()
    elif "--demo" in sys.argv:
        demo()
    else:
        menu()
