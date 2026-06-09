# primecount — l'algorithme de Gourdon pour π(x)

[![selftest](https://github.com/nicoa2701/primegourdon/actions/workflows/selftest.yml/badge.svg)](https://github.com/nicoa2701/primegourdon/actions/workflows/selftest.yml)

*[English version: README.md](README.md)*

Une implémentation C++20 *from scratch* de l'**algorithme de Xavier Gourdon**
pour la fonction de comptage des nombres premiers π(x) — le nombre de premiers
≤ x — calculée analytiquement, sans énumérer les premiers.

Elle repose sur l'identité de Gourdon :

```
π(x) = A − B + C + D + Φ₀ + Σ
```

Chacun des six termes est calculé indépendamment (et peut être évalué isolément
en ligne de commande), puis les termes sont combinés. La spécification
mathématique complète — chaque formule, borne de sommation, signe et piège
d'arithmétique entière — se trouve dans **[gourdon.md](gourdon.md)** ; le code en
est une réalisation fidèle et fortement optimisée.

## Points forts

- **Arithmétique entière exacte** de bout en bout ; bascule 128 bits au-delà de
  ~1,6·10²⁰ pour rester correct au-delà de 2⁶³.
- **Parallèle** (OpenMP) avec taille de segment adaptée au cache et à la bande
  passante.
- **Cribles segmentés wheel-30** pour les deux termes dominants (B et D), avec
  franchissement sans branche déroulé 8 voies et pré-crible SIMD.
- **Dispatch SIMD au runtime** (AVX2 / AVX-512) — un seul binaire tourne partout.
- **Micro-benchmark au configure** : choisit le backend de division 32/64 bits le
  plus rapide pour la machine de build.
- **Build phasé en mémoire** : la table π et la table de Möbius / plus-petit-
  facteur-premier ne coexistent jamais, divisant ~par deux le pic RSS sans coût
  en temps.
- **Garde-fou RAM adaptatif** : refuse en amont plutôt que de se faire
  OOM-killer en cours de route.

## Compilation

Nécessite CMake ≥ 3.16, un compilateur C++20 (GCC/Clang) et OpenMP.

```sh
./run.sh                 # configure + compile (Release)
./run.sh --selftest      # compile, puis lance la suite de tests
./run.sh 1e18 -v         # compile, puis exécute avec le détail par terme
```

`run.sh` est robuste au déplacement du dossier (il efface un cache CMake périmé
et reconfigure). Le binaire est placé à la racine du projet : `./primecount`.

### Options de compilation

Le code des feuilles faciles A/C peut utiliser un chemin de division 32 bits
protégé (`divl`) sur x86-64. Par défaut, CMake lance un micro-benchmark à la
configuration et l'active seulement s'il est plus rapide sur la machine de build.

```sh
./run.sh -DWITH_DIV32=ON --clean   # force le backend divl
./run.sh -DWITH_DIV32=OFF --clean  # force la division 64 bits classique
```

Les options de configuration sont gardées en cache par CMake ; utilise `--clean`
pour les changer. Pour une compilation purement locale, tu peux aussi remplacer
les flags Release dans `CMakeLists.txt` par `-march=native` afin de cibler
exactement ton CPU.

## Utilisation

```
Usage : primecount X [option]
  X                 entier ou notation scientifique (ex. 1e19)
  --Phi0            calcule seulement le terme Phi0
  --Sigma           calcule seulement le terme Sigma
  --AC              calcule seulement le terme A+C
  --B               calcule seulement le terme B
  --D               calcule seulement le terme D
  --perf            optimise pour la vitesse (défaut)
  --ram             optimise le pic RAM (~x^1/3) au détriment de la vitesse
  --auto            perf si le pic tient dans la RAM libre, sinon ram
  -v, --verbose     affiche chaque terme avec son propre chronométrage
  -t, --threads N   nombre de threads (défaut : tous les cœurs)
  --force           exécute même si le pic RSS estimé dépasse la RAM libre
  -h, --help        affiche l'aide
```

### Mémoire contre vitesse

Par défaut le chemin rapide est utilisé : le pic RSS croît en ~0,14·√x. Pour de
très grands x cela devient la contrainte limitante ; un **chemin bas-mémoire** est
donc disponible, dont le pic suit ~O(x^(1/3)), pour environ 1,15–1,2× le temps :

- `--perf` (défaut) — le plus rapide ; pic RSS ~√x.
- `--ram` — chemin bas-mémoire (pas de π-table en O(√x), table de Möbius/PPF
  compactée, C éparse) ; pic RSS ~x^(1/3), résultat bit-identique.
- `--auto` — exécute `--perf` si son pic estimé tient dans la RAM libre, sinon
  bascule sur `--ram` au lieu de refuser.

Ces modes pilotent le calcul complet de π(x) ; les flags par terme utilisent
toujours la construction par défaut.

Exemple :

```sh
$ ./primecount 1e18
24739954287740860  [46.410 s]

$ ./primecount 1e18 -v
RAM       available 4.94 GiB, estimated peak ~133.51 MiB, max x ~1.4e+21
build     DIV32 ON, AVX512 OFF, AVX2/BMI2/POPCNT on, threads 8
build   (shared ctx, built once)            [1.517 s]
Phi0  = 64014967544662                       [164 ms]
Sigma = 514634213323316                      [3 ms]
AC    = 9336325709491971                     [19.910 s]
B     = -9158014307746509                    [6.093 s]
D     = 23982993705127420                    [19.323 s]
pi(x) = 24739954287740860                    [45.493 s, +1.517 s build]
```

## Benchmark

Chemin rapide mono-thread sous 10¹¹ ; algorithme de Gourdon (tous cœurs) au-delà.
Temps wall-clock et **pic mémoire résidente** (`/usr/bin/time -v`, Maximum RSS).

**Machine :** Intel Core i5-9300HF (Coffee Lake, 4C / 8T, AVX2), 8 threads,
DIV32 activé automatiquement.

| x     | π(x)                  | temps     | pic RAM    |
|-------|-----------------------|-----------|------------|
| 10¹   | 4                     | < 1 ms    | 4,1 MiB    |
| 10²   | 25                    | < 1 ms    | 4,0 MiB    |
| 10³   | 168                   | < 1 ms    | 4,1 MiB    |
| 10⁴   | 1 229                 | < 1 ms    | 4,1 MiB    |
| 10⁵   | 9 592                 | < 1 ms    | 4,0 MiB    |
| 10⁶   | 78 498                | 2 ms      | 6,8 MiB    |
| 10⁷   | 664 579               | 2 ms      | 6,8 MiB    |
| 10⁸   | 5 761 455             | 2 ms      | 6,8 MiB    |
| 10⁹   | 50 847 534            | 3 ms      | 6,8 MiB    |
| 10¹⁰  | 455 052 511           | 4 ms      | 6,8 MiB    |
| 10¹¹  | 4 118 054 813         | 11 ms     | 6,8 MiB    |
| 10¹²  | 37 607 912 018        | 17 ms     | 8,4 MiB    |
| 10¹³  | 346 065 536 839       | 47 ms     | 9,3 MiB    |
| 10¹⁴  | 3 204 941 750 802     | 120 ms    | 10,7 MiB   |
| 10¹⁵  | 29 844 570 422 669    | 0,52 s    | 14,5 MiB   |
| 10¹⁶  | 279 238 341 033 925   | 2,21 s    | 24,9 MiB   |
| 10¹⁷  | 2 623 557 157 654 233 | 10,20 s   | 52,6 MiB   |
| 10¹⁸  | 24 739 954 287 740 860 | 46,41 s  | 125,0 MiB  |
| 10¹⁹  | 234 057 667 276 344 607 | 220,7 s | 321,7 MiB  |

(Chemin `--perf` par défaut.) Le chemin **`--ram`** échange ~1,15–1,2× le temps
contre un pic en O(x^(1/3)) ; pic RSS mesuré 25,8 / 48,6 / 108,8 / 270,1 MiB à
10¹⁶ / 10¹⁷ / 10¹⁸ / 10¹⁹ — il passe sous `--perf` vers 10¹⁷ et l'écart se creuse
avec x (−16 % à 10¹⁹).

Le temps suit **O(x^(2/3) / (log x)²)** (≈ ×4,7 par décade au-delà de 10¹⁴) ; le
pic RAM par défaut suit ≈ 0,14·√x, le pic `--ram` ≈ O(x^(1/3)).

## Correction

`./run.sh --selftest` vérifie, contre un oracle par crible segmenté exact :

- chaque terme (Φ₀, Σ, A, B, …) contre une implémentation brute indépendante ;
- l'assemblage complet pour x ∈ {10³ … 10⁸} ;
- **tous** les x de [1, 5000] contre le crible exact (attrape les bords
  off-by-one) ;
- la table connue π(10⁹)=50 847 534, π(10¹⁰)=455 052 511,
  π(10¹¹)=4 118 054 813.

Toutes les valeurs ci-dessus sont bit-exactes face aux références publiées.

## Limites

Validé exact jusqu'à **10²¹** (π(10²¹) correspond à la référence). Au-delà de
10²¹, certains accumulateurs de feuilles en int64 ne sont pas testés et peuvent
finir par déborder ; le programme affiche un avertissement. Le chemin de division
128 bits garde les *divisions* correctes au-delà, mais ce sont les accumulateurs
qui constituent le plafond pratique.

## Algorithme & références

La décomposition, les six termes et les subtilités d'arithmétique entière sont
documentés dans **[gourdon.md](gourdon.md)**. Sources principales :

- X. Gourdon, *Computation of π(x): improvements to the Meissel, Lehmer,
  Lagarias, Miller, Odlyzko, Deléglise and Rivat method*, 2001.
- M. Deléglise, J. Rivat, *Computing π(x): The Meissel–Lehmer–Lagarias–Miller–
  Odlyzko Method*, Math. Comp. 65 (1996).
- D. B. Staple, *The combinatorial algorithm for computing π(x)*, 2015.

Pour l'implémentation de référence à l'état de l'art, voir le
[primecount](https://github.com/kimwalisch/primecount) de Kim Walisch.

## Licence

BSD 2-Clause. Voir [LICENSE](LICENSE).
