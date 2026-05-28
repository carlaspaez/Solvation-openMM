import os
import json
import re

# ===== METODE 1 =====
def metode_1():
    """
    Extrae números de archivos error y sortid en outputs_PBC-ok/cluster_PBC,
    busca sus jobIDs en resultats/costcompu.json y guarda los valores Elapsed.
    """
    # Paso 1: Buscar archivos error y sortid, extraer números
    cluster_path = "outputs_PBC-ok/cluster_PBC"
    numeros = set()
    
    if os.path.exists(cluster_path):
        for filename in os.listdir(cluster_path):
            # Buscar archivos que empiezan con "error" o "sortid" seguidos de números
            match = re.match(r'(error|sortid)_?(\d+)', filename)
            if match:
                numero = match.group(2)
                numeros.add(numero)
    
    numeros = sorted(list(numeros))
    print(f"Números encontrados: {len(numeros)}")
    print(f"Primeros números: {numeros[:10]}")
    
    # Paso 2: Buscar jobIDs en resultats/costcompu.json
    json_path = "resultats/costcompu.json"
    valores_elapsed = []
    valores_detallados = []
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        jobs = data.get("jobs", [])
        jobs_per_id = {str(entry.get("JobID")): entry for entry in jobs}
        
        # Buscar jobID para cada número
        for numero in numeros:
            entry = jobs_per_id.get(numero)
            if entry is None:
                continue

            elapsed = entry.get("Elapsed")
            if elapsed is not None:
                valores_elapsed.append(elapsed)
                valores_detallados.append({
                    "numero": numero,
                    "JobID": entry.get("JobID"),
                    "Elapsed": elapsed
                })
    
    print(f"Elapsed encontrados: {len(valores_elapsed)}")
    print(f"Primeros Elapsed: {valores_elapsed[:10]}")
    print(f"Primeros detalles: {valores_detallados[:10]}")
    return valores_elapsed

if __name__ == "__main__":
    metode_1()
