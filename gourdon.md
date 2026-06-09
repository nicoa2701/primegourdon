# L'algorithme de Xavier Gourdon pour $\pi(x)$ — formules détaillées

> Document mathématique destiné à une implémentation « from scratch ».
> Aucune ligne de code : uniquement les définitions, les formules exactes,
> les bornes de sommation, les signes et l'ordre des opérations.
>
> Référence d'origine : Xavier Gourdon, *Computation of $\pi(x)$ : improvements
> to the Meissel, Lehmer, Lagarias, Miller, Odlyzko, Deléglise and Rivat method*,
> 15 février 2001. L'algorithme est une amélioration de Deléglise–Rivat (D–R),
> lui-même issu de Lagarias–Miller–Odlyzko (LMO) et de Meissel–Lehmer.

---

## 0. Conventions et notations

| Symbole | Définition |
|---|---|
| $\pi(t)$ | nombre de premiers $\le t$ |
| $p_b$ | le $b$-ième premier ($p_1 = 2,\ p_2 = 3,\dots$) |
| $P^-(n)$ | plus petit facteur premier de $n$ |
| $P^+(n)$ | plus grand facteur premier de $n$ |
| $\mu(n)$ | fonction de Möbius |
| $\lfloor\cdot\rfloor$ | partie entière (toutes les divisions sont **entières**) |
| $\varphi(t,b)$ | **fonction de crible partielle** (Legendre) : nombre d'entiers de $[1,t]$ qui ne sont divisibles par aucun des $b$ premiers premiers $p_1,\dots,p_b$ |

> ⚠️ Toute l'arithmétique se fait sur des entiers. Pour $x \gtrsim 10^{19}$ il faut
> un type 128 bits pour $x$ et les produits intermédiaires.

### Identité fondamentale de Gourdon

$$\boxed{\;\pi(x) = A - B + C + D + \Phi_0 + \Sigma\;}$$

Les six termes sont calculés indépendamment puis combinés à la toute fin.
Aucun n'a besoin du résultat d'un autre : tu peux les vérifier un par un.

Idée d'ensemble (pour comprendre, pas nécessaire au calcul) :
- Meissel : $\pi(x) = \varphi(x,a) + a - 1 - P_2(x,a)$.
- Gourdon décompose $\varphi(x,a)$ via le critère de Möbius en
  **feuilles ordinaires** ($\Phi_0$), **feuilles spéciales** ($A$, $C$, $D$)
  et **termes triviaux** ($\Sigma$, les 7 sommes $\Sigma_0..\Sigma_6$).
- $B$ est la partie « non triviale » de $P_2(x,a)$.

---

## 1. Les paramètres (à calculer en premier)

Tout dépend de quatre quantités : $y$, $z$, $x^\star$ et $k$, déterminées par
deux facteurs de réglage $\alpha_y \ge 1$ et $\alpha_z \ge 1$.

### 1.1 $y$ et $z$

$$
y = \big\lfloor \alpha_y \cdot x^{1/3} \big\rfloor,
\qquad
z = \big\lfloor \alpha_z \cdot y \big\rfloor = \big\lfloor \alpha_y\,\alpha_z\, x^{1/3} \big\rfloor .
$$

Contraintes (à imposer par bornage après calcul) :

$$
x^{1/3} < y < x^{1/2},
\qquad
y \le z < x^{1/2}.
$$

Concrètement : $y \leftarrow \max(y,\ \lfloor x^{1/3}\rfloor + 1)$, puis
$y \leftarrow \min(y,\ \lfloor x^{1/2}\rfloor - 1)$ ; idem pour $z$ avec
$z \leftarrow \max(z, y)$ et $z \leftarrow \min(z, \lfloor x^{1/2}\rfloor - 1)$.

### 1.2 $x^\star$

$$
\boxed{\;x^\star = \max\!\Big(\big\lfloor x^{1/4}\big\rfloor,\ \big\lceil x / y^2 \big\rceil\Big)\;}
$$

bornée ensuite par $x^\star \le y$ et $x^\star \le \sqrt{x/y}$ (et $\ge 1$).

> Note d'implémentation : il faut **arrondir $x/y^2$ vers le haut** ($\lceil\cdot\rceil$).
> Sans cela, de nombreuses petites valeurs ($x < 2000$) sont mal calculées.

$x^\star$ est la frontière entre les feuilles « faciles » traitées dans
$A$ / $C_2$ et les feuilles « dures » de $D$ ; c'est aussi une borne pivot dans $\Sigma$.

### 1.3 $k$ (la petite constante, notée $c$ chez Deléglise–Rivat)

$$
k = \pi\!\big(x^{1/4}\big),
\qquad
\text{plafonné : } k \le \pi\!\big(\min(x^\star,\ \sqrt{x/y})\big),
\qquad
k \le k_{\max}.
$$

$k_{\max}$ est le plus grand $a$ pour lequel tu sais évaluer $\varphi(t,a)$ par
**table périodique** (voir §2). En pratique $k_{\max} \in \{7, 8\}$
(le produit $p_1\cdots p_8 = 9\,699\,690$ tient en mémoire ; $p_1\cdots p_7 = 510\,510$).

### 1.4 Choix de $\alpha_y$, $\alpha_z$ (heuristique de réglage)

Ce ne sont que des facteurs de performance ; **n'importe quelle valeur valide
donne le bon résultat**, seul le temps de calcul change. Heuristique éprouvée :

Pose $L = \ln x$ et calcule un produit $\alpha_y\alpha_z =: \alpha_{yz}$ :

$$
\alpha_{yz} =
\begin{cases}
0.078173\,L + 1 & \text{si } x \le 10^{11},\\[4pt]
0.00526934\,L^3 - 0.495545\,L^2 + 16.5791\,L - 183.836 & \text{sinon.}
\end{cases}
$$

Puis : $\alpha_z \approx \min\!\big(\alpha_{yz}/5,\ 1.5\big)$ borné dans $[1, \alpha_{yz}]$,
et $\alpha_y = \alpha_{yz}/\alpha_z$. Borne finale : $\alpha_{yz} \le x^{1/6}$
(pour garantir $y < x^{1/2}$).

Pour débuter, $\alpha_y = \alpha_z = 1$ (donc $y = z = x^{1/3}$) **fonctionne**
et simplifie tout : commence par là pour valider la correction, puis optimise.

---

## 2. La fonction de crible partielle $\varphi(t,b)$

Brique de base de $\Phi_0$, $C$ et $D$. Définition :
$\varphi(t,b) = \#\{\,1 \le n \le t : p_1\nmid n,\dots,p_b\nmid n\,\}$.

Récurrence de Legendre :

$$
\varphi(t,b) = \varphi(t,b-1) - \varphi\!\big(\lfloor t/p_b\rfloor,\ b-1\big),
\qquad \varphi(t,0) = \lfloor t\rfloor .
$$

### 2.1 Cas $b \le k$ : $\varphi_k(t) := \varphi(t,k)$ par table (« phi_tiny »)

$\varphi(\cdot,b)$ est **périodique** de période $P_b = p_1 p_2\cdots p_b$ :

$$
\varphi(t,b) = \big\lfloor t / P_b \big\rfloor \cdot \Phi(P_b, b) \;+\; \Phi(t \bmod P_b,\ b),
$$

où $\Phi(P_b,b) = \prod_{i=1}^{b}(p_i - 1)$ (totient du produit) et
$\Phi(r,b)$ pour $0 \le r < P_b$ est **précalculé** dans un tableau de taille $P_b$.
C'est ce qui rend $\varphi_k$ évaluable en $O(1)$. C'est pour cela que $k$ doit rester petit.

### 2.2 Cas $b$ « facile » : l'identité clé

Lorsque $p_b \le \sqrt{t} < p_{b+1}$ (c.-à-d. tous les premiers $> p_b$ qui sont
$\le t$ sont eux-mêmes premiers et $\le t$), on a l'identité **fondamentale**
utilisée dans $A$, $C$, $D$ :

$$
\boxed{\;\varphi(t,\,b-1) = \pi(t) - (b-1) + 1 = \pi(t) - b + 2\;}
$$

valable quand $t < p_b^2$. Elle transforme un crible coûteux en une simple
lecture de table $\pi$. Les feuilles qui la satisfont sont dites **faciles** ;
celles qui ne la satisfont pas sont **dures** (calculées par crible segmenté, §7).

---

## 3. $\Phi_0$ — feuilles ordinaires

> Attention : la formule de $\Phi_0$ donnée page 7 du papier de Gourdon est
> **erronée** ; la bonne se trouve page 3.

$$
\boxed{\;
\Phi_0 = \sum_{\substack{n \ge 1,\ n \text{ sans facteur carré}\\ P^-(n) > p_k,\ \ P^+(n) \le y,\ \ n \le z}}
\mu(n)\,\varphi\!\Big(\big\lfloor x/n\big\rfloor,\ k\Big)
\;}
$$

Domaine de sommation : les entiers $n$
- **sans facteur carré** (squarefree),
- premiers à $p_1,\dots,p_k$ (c.-à-d. $P^-(n) > p_k$),
- de plus grand facteur premier $\le y$,
- avec $n \le z$.

Le terme $n=1$ donne $\mu(1)\,\varphi_k(x) = \varphi(x,k)$.

### Recette de calcul

1. Initialiser $\Phi_0 \leftarrow \varphi(x,k)$ (terme $n=1$).
2. Pour chaque indice $b = k+1,\,k+2,\dots,\pi(y)$ (premier $p_b$ servant de **plus petit** facteur) :
   - retrancher $\varphi\!\big(\lfloor x/p_b\rfloor,\ k\big)$ (cas $n = p_b$, $\mu = -1$) ;
   - explorer **récursivement** tous les produits sans facteur carré
     $n = p_b \cdot p_{i_1}\cdots p_{i_r}$ avec
     $b < i_1 < i_2 < \dots$ (facteurs strictement croissants donc distincts),
     $p_{i_r} \le y$, et $n \le z$ ; à chaque nœud ajouter
     $\mu(n)\,\varphi(\lfloor x/n\rfloor, k)$, le signe $\mu$ alternant
     ($+$ pour un nombre pair de facteurs, $-$ pour un nombre impair).

   La récursion s'arrête dès que $n \cdot p_{\text{suivant}} > z$.

Coût : $O(z)$ en temps, mémoire $O(y/\log y)$ (la liste des premiers $\le y$).

---

## 4. $\Sigma$ — les sept termes triviaux $\Sigma_0,\dots,\Sigma_6$

On pose, **une fois pour toutes**, quatre compteurs de premiers :

$$
a = \pi(y),\qquad
b = \pi\!\big(x^{1/3}\big),\qquad
c = \pi\!\big(\sqrt{x/y}\big),\qquad
d = \pi\!\big(x^\star\big).
$$

(Ici $b,c,d$ sont des scalaires, à ne pas confondre avec les indices de boucle.)

$$
\Sigma = \Sigma_0 + \Sigma_1 + \Sigma_2 + \Sigma_3 + \Sigma_4 + \Sigma_5 + \Sigma_6 .
$$

### Termes purement combinatoires (formules fermées)

$$
\Sigma_0 = a - 1 + \frac{\pi(\sqrt x)\,\big(\pi(\sqrt x) - 1\big)}{2} - \frac{a(a-1)}{2}
$$

$$
\Sigma_1 = \frac{(a-b)\,(a-b-1)}{2}
$$

$$
\Sigma_2 = a\left(\,b - c - \frac{c(c-3)}{2} + \frac{d(d-3)}{2}\,\right)
$$

$$
\Sigma_3 = \frac{b(b-1)(2b-1)}{6} - b \;-\; \left(\frac{d(d-1)(2d-1)}{6} - d\right)
$$

> Toutes les divisions ci-dessus tombent juste sur les entiers (sommes de
> Faulhaber) : $n(n-1)/2$, $\sum i^2 = n(n+1)(2n+1)/6$, etc.

### Termes avec balayage de premiers

$$
\Sigma_4 = a \sum_{\substack{p \text{ premier}\\ x^\star < p \le \sqrt{x/y}}}
\pi\!\left(\Big\lfloor \frac{x}{p\,y} \Big\rfloor\right)
$$

$$
\Sigma_5 = \sum_{\substack{p \text{ premier}\\ \sqrt{x/y} < p \le x^{1/3}}}
\pi\!\left(\Big\lfloor \frac{x}{p^2} \Big\rfloor\right)
$$

$$
\Sigma_6 = -\sum_{\substack{p \text{ premier}\\ x^\star < p \le x^{1/3}}}
\Big(\pi\big(\big\lfloor\sqrt{\lfloor x/p\rfloor}\big\rfloor\big)\Big)^{2}
$$

> ⚠️ **$\Sigma_6$ — piège d'entiers.** Dans le papier, $\Sigma_6$ s'écrit
> $\sum \pi(\sqrt{x}/\sqrt{p})^2$. En arithmétique entière cette écriture donne
> des résultats **faux** : il faut impérativement calculer
> $\big(\pi(\lfloor\sqrt{\lfloor x/p\rfloor}\rfloor)\big)^2$,
> c'est-à-dire racine de la **division entière** $\lfloor x/p\rfloor$ d'abord.

### Recette $\Sigma_4 + \Sigma_5 + \Sigma_6$ (une seule boucle)

Construire une table $\pi[\,\cdot\,]$ jusqu'à
$\max\!\big(\lfloor x/(x^\star y)\rfloor,\ y,\ \lfloor\sqrt{x/x^\star}\rfloor\big)$.
Puis parcourir les premiers $p$ de $x^\star + 1$ jusqu'à $x^{1/3}$ :
- si $p \le \sqrt{x/y}$ : ajouter $\pi[\lfloor x/(p\,y)\rfloor]$ à un accumulateur $S_4$ ;
- sinon : ajouter $\pi[\lfloor x/p^2\rfloor]$ à $S_5$ ;
- dans **tous** les cas : ajouter $\big(\pi[\lfloor\sqrt{\lfloor x/p\rfloor}\rfloor]\big)^2$ à $S_6$.

Résultat : $\Sigma_4 = a\,S_4$, $\Sigma_5 = S_5$, $\Sigma_6 = -S_6$.

Coût : $\Sigma_0$ en $O(\sqrt x)$ (le $\pi(\sqrt x)$), le reste en $O(y)$.

---

## 5. $A$ — feuilles spéciales faciles (deux premiers, leaf $\ge$ pivot)

$$
\boxed{\;
A = \sum_{\substack{b\ :\ x^\star < p_b \le x^{1/3}}}
\ \sum_{\substack{i\ :\ p_b < p_i \le \sqrt{x/p_b}}}
w(b,i)\ \cdot\ \pi\!\left(\Big\lfloor \frac{x}{p_b\,p_i} \Big\rfloor\right)
\;}
$$

avec le **poids**

$$
w(b,i) =
\begin{cases}
1 & \text{si } \big\lfloor x/(p_b p_i)\big\rfloor \ge y
   \quad\Longleftrightarrow\quad p_i \le \big\lfloor x/(p_b\,y)\big\rfloor,\\[4pt]
2 & \text{si } \big\lfloor x/(p_b p_i)\big\rfloor < y.
\end{cases}
$$

### Recette

Pour chaque premier $p_b$ avec $x^\star < p_b \le x^{1/3}$, pose $x_p = \lfloor x/p_b\rfloor$
et $s = \lfloor\sqrt{x_p}\rfloor$. Le second premier $q = p_i$ vérifie $p_b < q \le s$.
- Tant que $q \le \lfloor x_p / y\rfloor$ : ajouter $1\cdot\pi(\lfloor x_p/q\rfloor)$.
- Au-delà (jusqu'à $q \le s$) : ajouter $2\cdot\pi(\lfloor x_p/q\rfloor)$.

La valeur $\lfloor x/(p_b p_i)\rfloor$ peut atteindre $\sqrt x$ ; d'où l'usage
d'une **table $\pi$ segmentée** allant jusqu'à $x^{1/2}$ (mémoire ramenée de
$O(x^{1/2})$ à $O(x^{1/4})$ par segmentation, mais ce n'est qu'une optimisation).

---

## 6. $C$ — feuilles spéciales faciles (leaf $\le$ pivot) : $C = C_1 + C_2$

### 6.1 $C_2$ — deux premiers

$$
\boxed{\;
C_2 = \sum_{\substack{b\ :\ \pi(\sqrt z) < p_b \le x^\star}}
\ \sum_{\substack{q \text{ premier}\\ p_b < q \le y,\ \ p_b q > z}}
\Big(\pi\!\big(\lfloor x/(p_b q)\rfloor\big) - b + 2\Big)
\;}
$$

Le terme $\pi(\cdot) - b + 2 = \varphi(\cdot,\,b-1)$ est l'identité §2.2
(feuille facile car $x/(p_b q) < p_b^2$ ici).

Pour chaque $p_b$, le second premier $q$ parcourt $(\,m_{\min},\,m_{\max}\,]$ avec
$$
m_{\max} = \min\!\Big(\big\lfloor x/(p_b\cdot\text{low})\big\rfloor,\ \big\lfloor x_p/p_b\big\rfloor,\ y\Big),
\qquad
m_{\min} = \max\!\Big(\big\lfloor x/(p_b\cdot\text{high})\big\rfloor,\ \big\lfloor x_p/p_b^2\big\rfloor,\ p_b\Big),
$$
où $[\text{low},\text{high})$ est le segment courant de la table $\pi$ segmentée et $x_p=\lfloor x/p_b\rfloor$.
(En version non segmentée : $\text{low}=1$, $\text{high}=x^{1/2}$.)

### 6.2 $C_1$ — un premier × un nombre sans facteur carré

$$
\boxed{\;
C_1 = \sum_{\substack{b\ :\ (x/z)^{1/3} < p_b \le \sqrt z}}
\ \sum_{m}
\mu(m)\ \Big(\pi\!\big(\lfloor x/(p_b m)\rfloor\big) - b + 2\Big)
\;}
$$

où $m$ parcourt les entiers **sans facteur carré** tels que :
- $P^-(m) > p_b$ (premiers à $p_1,\dots,p_b$ — facteurs strictement $> p_b$),
- $P^+(m) \le y$,
- $z/p_b < m \le m_{\max}$ avec la feuille restant « facile »
  (c.-à-d. $x/(p_b m)$ assez petit).

C'est la **même récursion sur les squarefree** que pour $\Phi_0$ (§3), mais
on additionne $\mu(m)\cdot\big(\pi(\lfloor x/(p_b m)\rfloor) - b + 2\big)$ au lieu de
$\mu(n)\,\varphi_k$.

> $C_1$ n'est pas segmenté et exige des synchronisations fréquentes :
> c'est la partie qui passe mal à l'échelle au-delà de $\sim 10^{23}$.
> Augmenter $\alpha_z$ réduit le travail de $C_1$.

Dans l'implémentation de référence, $A$, $C_1$ et $C_2$ partagent **la même
table $\pi$ segmentée** parcourue de $0$ à $x^{1/2}$, d'où leur fusion ; ce n'est
qu'une optimisation, tu peux les calculer séparément.

---

## 7. $D$ — feuilles spéciales dures : $D = D_1 + D_2$

Ce sont les feuilles où l'identité facile §2.2 **ne** s'applique **pas** :
$\varphi(\lfloor x/(p\cdot m)\rfloor,\ b-1)$ doit être obtenu par **crible segmenté**.
C'est le terme le plus coûteux ($O(x^{2/3}/(\log x)^2)$). Les feuilles dures
sont celles dont la valeur tombe dans $[1, z]$.

### 7.1 $D_1$ — un premier × un squarefree

$$
\boxed{\;
D_1 = -\sum_{\substack{b\ :\ k < b \le \pi(\sqrt z)}}
\ \sum_{m}
\mu(m)\ \varphi\!\Big(\big\lfloor x/(p_b m)\big\rfloor,\ b-1\Big)
\;}
$$

où $m$ parcourt les entiers **sans facteur carré** vérifiant :
- $P^-(m) > p_b$ (least prime factor $> p_b$),
- $P^+(m) \le y$ et $m \le z$,
- feuille **dure** : $\big\lfloor x/(p_b m)\big\rfloor \le z$,
  soit l'intervalle $m \in (m_{\min}, m_{\max}]$ avec
  $$
  m_{\min} = \max\!\Big(\big\lfloor x/(p_b\,\text{high})\big\rfloor,\ \big\lfloor z/p_b\big\rfloor\Big),
  \qquad
  m_{\max} = \min\!\Big(\big\lfloor x/p_b^3\big\rfloor,\ \big\lfloor x/(p_b\,\text{low})\big\rfloor,\ z\Big).
  $$

### 7.2 $D_2$ — deux premiers

$$
\boxed{\;
D_2 = \sum_{\substack{b\ :\ \pi(\sqrt z) < p_b \le x^\star}}
\ \sum_{\substack{q \text{ premier}\\ p_b < q \le y,\ \ \lfloor x/(p_b q)\rfloor \le z}}
\varphi\!\Big(\big\lfloor x/(p_b q)\big\rfloor,\ b-1\Big)
\;}
$$

(Le signe est $+$ car $m=q$ est un seul premier, $\mu(q) = -1$, et $-\mu(q)=+1$.)

### 7.3 Recette — crible segmenté de $\varphi(\cdot,\,b-1)$

On ne recalcule pas $\varphi$ de zéro à chaque feuille : on **maintient** un crible.

Découper $[1,\ \lfloor x/z\rfloor]$ en segments $[\text{low},\text{high})$. Pour chaque segment :

1. **Pré-cribler** : marquer dans le segment les multiples des premiers
   $p_1,\dots,p_{k}$ (ou jusqu'à $p_{\min_b - 1}$). Maintenir un tableau de
   comptage permettant `count(n)` = nombre d'entiers non barrés dans $[\text{low}, n]$.
2. Entretenir, pour chaque $b$, une valeur courante $\varphi[b]$ =
   $\varphi(\text{low}-1,\ b-1)$ (le compte cumulé jusqu'au début du segment).
3. Parcourir $b$ de $k+1$ jusqu'à $\pi(\sqrt z)$ (**partie $D_1$**) :
   - pour chaque squarefree $m$ admissible (cf. §7.1) avec
     $P^-(m) > p_b$, lire la valeur de la feuille $v = \lfloor x/(p_b m)\rfloor$ ;
   - si $v$ tombe dans le segment courant :
     $\varphi(v, b-1) = \varphi[b] + \text{count}(v)$ ;
   - accumuler $-\mu(m)\cdot\varphi(v,b-1)$.
   - après avoir traité $b$ : mettre à jour $\varphi[b] \mathrel{+}= (\text{total non barré du segment})$,
     puis **barrer** dans le crible tous les multiples de $p_b$ (passage de
     $\varphi(\cdot,b-1)$ à $\varphi(\cdot,b)$ pour le segment suivant).
4. Puis $b$ de $\pi(\sqrt z)+1$ jusqu'à $\pi(x^\star)$ (**partie $D_2$**) :
   même mécanique mais $m = q$ parcourt les **premiers** $p_b < q \le y$ ;
   accumuler $+\varphi(\lfloor x/(p_b q)\rfloor,\ b-1)$.

Le repérage des squarefree $m$ et de leur $\mu$ se fait via une table de facteurs
(plus petit facteur premier + valeur de Möbius) précalculée sur $[1, z]$ — c'est
l'analogue exact des « hard special leaves » de Deléglise–Rivat.

---

## 8. $B$ — partie non triviale de $P_2(x,a)$

$$
\boxed{\;
B = \sum_{\substack{i\ :\ \pi(y) < p_i \le \sqrt x}}
\pi\!\left(\Big\lfloor \frac{x}{p_i} \Big\rfloor\right)
\;}
$$

$B$ compte (avec $P_2$) les entiers $\le x$ ayant exactement deux facteurs
premiers, tous deux $> p_{\pi(y)}$. Complexité $O(n \log\log n)$, mémoire
$O(\sqrt n)$ avec $n = x/y$.

### Recette

Parcourir les premiers $p_i$ de $\sqrt x$ vers le bas (ou de $y$ vers $\sqrt x$),
et sommer $\pi(\lfloor x/p_i\rfloor)$. Comme $\lfloor x/p_i\rfloor$ peut aller
jusqu'à $x/y$, on calcule ces $\pi$ par **crible segmenté** sur $[y,\ x/y]$ :
la première valeur peut être obtenue par un appel direct à une fonction de
comptage de premiers, les suivantes par avancée du crible.

---

## 9. Assemblage final

1. Calcule $\alpha_y,\alpha_z \Rightarrow y, z, x^\star, k$ (§1).
2. Calcule indépendamment :
   $\Sigma$ (§4), $\Phi_0$ (§3), $A$ (§5), $C = C_1 + C_2$ (§6),
   $D = D_1 + D_2$ (§7), $B$ (§8).
3. Combine :
$$
\pi(x) = A - B + C + D + \Phi_0 + \Sigma .
$$

> Ordre conseillé d'exécution (charge croissante CPU/mémoire) :
> $\Sigma$, puis $\Phi_0$, puis $A{+}C$, puis $B$, puis $D$.
> Cela ne change pas le résultat, seulement le profil de performance.

### Complexité globale

$$
\text{temps } O\!\Big(\frac{x^{2/3}}{(\log x)^2}\Big),
\qquad
\text{mémoire } O\!\big(x^{1/3}\,(\log x)^3\big).
$$

---

## 10. Stratégie de validation (indispensable)

1. **Cas dégénéré $\alpha_y=\alpha_z=1$** : commence par $y=z=x^{1/3}$.
   Beaucoup de sommes deviennent vides ($\Sigma_4,\Sigma_5,\dots$), c'est plus facile à déboguer.
2. **Comparaison terme à terme** : vérifie chaque terme ($A,B,C,D,\Phi_0,\Sigma$)
   séparément contre une implémentation de référence sur de petits $x$.
3. **Petites valeurs connues** :
   $\pi(10) = 4$, $\pi(100) = 25$, $\pi(10^3) = 168$, $\pi(10^4) = 1229$,
   $\pi(10^6) = 78498$, $\pi(10^8) = 5761455$, $\pi(10^9) = 50847534$,
   $\pi(10^{10}) = 455052511$.
4. **Test de cohérence** : $\pi(x) \approx \operatorname{Li}(x)$ ;
   un écart énorme signale un bug de signe ou de borne (les pièges les plus
   fréquents : $\Sigma_6$ §4, l'arrondi de $x^\star$ §1.2, et les divisions
   entières dans les bornes de boucle).
5. **Pièges récurrents** :
   - mélanger division entière et division flottante dans une borne ;
   - oublier le « $-b+2$ » dans $C_1$, $C_2$ (identité §2.2) ;
   - mauvais signe global de $D$ (c'est $-\mu(m)$) ou de $\Sigma_6$ ;
   - feuille comptée deux fois entre $A$, $C$ et $D$ (vérifier que les
     intervalles en $p_b$ — $(x^\star, x^{1/3}]$ pour $A$, $(\sqrt z, x^\star]$
     pour $C_2$/$D_2$, $((x/z)^{1/3}, \sqrt z]$ pour $C_1$, $(k, \sqrt z]$ pour
     $D_1$ — sont disjoints et couvrants comme prévu).

---

## 11. Carte des intervalles en $p_b$ (vue synthétique)

```
premiers : p ↑  2 ........ x^(1/4)=k ..... (x/z)^{1/3} ... √z ...... x* ...... x^(1/3) ...... √(x/y) ... y ........ √x
                 │            │              │            │         │           │              │        │          │
Φ0 (P^-(n)>p_k) : ───────────►(facteurs des n, ≤ y)───────────────────────────────────────────────────►
D1 (1 prem × sf): k<b≤π(√z)   ├───────────────────────────►
C1 (1 prem × sf):                            (x/z)^{1/3}<b≤π(√z) ├────►
C2 (2 premiers) :                                                 √z<b≤π(x*) ├────►
D2 (2 premiers) :                                                 √z<b≤π(x*) ├────►
A  (2 premiers) :                                                            x*<b≤π(x^(1/3)) ├──────►
Σ4 (balayage p) :                                                            x*<p≤√(x/y)
Σ5 (balayage p) :                                                                         √(x/y)<p≤x^(1/3)
Σ6 (balayage p) :                                                            x*<p≤x^(1/3)
B  (balayage p) :                                                                                        y<p≤√x
```

---

### Sources

- X. Gourdon, *Computation of $\pi(x)$ : improvements to the Meissel, Lehmer,
  Lagarias, Miller, Odlyzko, Deléglise and Rivat method*, 2001.
- M. Deléglise, J. Rivat, *Computing $\pi(x)$: The Meissel, Lehmer, Lagarias,
  Miller, Odlyzko Method*, Math. Comp. 65 (1996), 235–245.
- J. C. Lagarias, V. S. Miller, A. M. Odlyzko, *Computing $\pi(x)$: The
  Meissel-Lehmer method*, Math. Comp. 44 (1985), 537–560.
- D. B. Staple, *The combinatorial algorithm for computing $\pi(x)$*, M.Sc.
  thesis, Dalhousie University, 2015 (récursion sur les squarefree, §2.2 du mémoire).
- T. Oliveira e Silva, *Computing $\pi(x)$: the combinatorial method*,
  Revista do DETUA 4 (6), 2006, 759–768.

