## Table of Contents
2. [La géométrie de la vision stéréo](#1-la-géométrie-de-la-vision-stéréo)
   - 2.1 [Modèle de caméra projective](#11-modèle-de-caméra-projective)
   - 2.2 [La baseline stéréo et la disparité](#12-la-baseline-stéréo-et-la-disparité)
3. [Détection et appariement de caractéristiques avec ORB](#2-détection-et-appariement-de-caractéristiques-avec-orb)
   - 3.1 [ORB : Oriented FAST and Rotated BRIEF](#21-orb--oriented-fast-and-rotated-brief)
   - 3.2 [Espace d'échelle et détection multi-échelle](#22-espace-déchelle-et-détection-multi-échelle)
   - 3.3 [Appariement par distance de Hamming](#23-appariement-par-distance-de-hamming)
4. [Géométrie épipolaire et matrice fondamentale](#3-géométrie-épipolaire-et-matrice-fondamentale)
   - 4.1 [Contrainte épipolaire](#31-contrainte-épipolaire)
   - 4.2 [Algorithme des 8 points (normalisé)](#32-algorithme-des-8-points-normalisé)
   - 4.3 [RANSAC pour une estimation robuste](#33-ransac-pour-une-estimation-robuste)
5. [Rectification stéréo non calibrée](#4-rectification-stéréo-non-calibrée)
   - 5.1 [But de la rectification](#41-but-de-la-rectification)
   - 5.2 [Algorithme de Hartley](#42-algorithme-de-hartley)
   - 5.3 [Détection automatique de la rectification](#43-détection-automatique-de-la-rectification)
6. [Semi-Global Block Matching (SGBM)](#5-semi-global-block-matching-sgbm)
   - 6.1 [Le problème de l'estimation de disparité](#51-le-problème-de-lestimation-de-disparité)
   - 6.2 [Coût de block matching](#52-coût-de-block-matching)
   - 6.3 [Énergie du semi-global matching](#53-énergie-du-semi-global-matching)
   - 6.4 [Programmation dynamique le long de trajectoires](#54-programmation-dynamique-le-long-de-trajectoires)
   - 6.5 [Post-traitement : unicité et filtrage des speckles](#55-post-traitement--unicité-et-filtrage-des-speckles)
7. [Carte de profondeur et reconstruction 3D](#6-carte-de-profondeur-et-reconstruction-3d)
   - 7.1 [De la disparité à la profondeur](#61-de-la-disparité-à-la-profondeur)
   - 7.2 [Rétroprojection 3D](#62-rétroprojection-3d)
   - 7.3 [Suppression des valeurs aberrantes et sous-échantillonnage](#63-suppression-des-valeurs-aberrantes-et-sous-échantillonnage)
8. [Résumé du pipeline](#7-résumé-du-pipeline)
9. [Référence des paramètres](#8-référence-des-paramètres)
10. [Références](#9-références)



## 1. La géométrie de la vision stéréo

### 1.1 Modèle de caméra projective

Une caméra projette un point 3D $\mathbf{P} = (X, Y, Z)^\top$ du monde vers un pixel 2D $\mathbf{p} = (u, v)^\top$ au moyen de la **projection en perspective** (modèle de sténopé) :

$$u = f \frac{X}{Z} + c_x, \qquad v = f \frac{Y}{Z} + c_y$$

où $f$ est la focale (en pixels) et $(c_x, c_y)$ est le point principal (généralement le centre de l'image). En coordonnées homogènes, cela s'écrit de manière compacte :

$$\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = \mathbf{K} \begin{pmatrix} X \\ Y \\ Z \end{pmatrix}, \qquad \mathbf{K} = \begin{pmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{pmatrix}$$

La matrice $\mathbf{K}$ est la **matrice de calibration intrinsèque**. Le scalaire $\lambda = Z$ est la profondeur perspective, introduite afin de rendre la projection linéaire en coordonnées homogènes.

### 1.2 La baseline stéréo et la disparité

Dans un **système stéréo rectifié**, les deux caméras partagent la même focale $f$ et leurs axes optiques sont parallèles. La caméra de droite est décalée d'une **baseline** $b$ suivant l'axe $X$. Dans ces conditions, un point 3D $\mathbf{P} = (X, Y, Z)^\top$ se projette au pixel $(u_L, v)$ dans l'image gauche et au pixel $(u_R, v)$ dans l'image droite ; point crucial, les coordonnées $v$ sont **identiques** après rectification.

La **disparité** est définie par :

$$d = u_L - u_R$$

En combinant les équations de projection des deux caméras, on obtient la relation fondamentale **profondeur-disparité** :

$$Z = \frac{f \cdot b}{d}$$

Cette relation inverse est centrale en vision stéréo : les grandes disparités correspondent aux objets proches, les petites disparités aux objets éloignés. Une fois $d$ connu pour chaque pixel, la position 3D complète s'obtient par rétroprojection :

$$X = \frac{(u_L - c_x) \cdot Z}{f}, \qquad Y = \frac{(v - c_y) \cdot Z}{f}$$

---

## 2. Détection et appariement de caractéristiques avec ORB

Avant de calculer une carte de disparité dense, le pipeline doit soit vérifier que la paire d'images est déjà rectifiée, soit estimer les transformations de rectification. Ces deux tâches nécessitent de trouver des **correspondances de points** entre les deux images, c'est-à-dire des paires de pixels $(\mathbf{p}_L, \mathbf{p}_R)$ représentant le même point 3D de la scène.

### 2.1 ORB : Oriented FAST and Rotated BRIEF

ORB (Rublee et al., 2011) est un descripteur binaire conçu pour être rapide, économe en mémoire et robuste aux changements de point de vue et d'illumination. Il combine deux briques de base :

**Détecteur de points d'intérêt FAST.** FAST (Features from Accelerated Segment Test) déclare qu'un pixel $p$ est un coin s'il existe un arc contigu d'au moins 9 pixels (parmi 16 placés sur un cercle de Bresenham de rayon 3) qui sont tous soit plus lumineux que $I_p + t$, soit plus sombres que $I_p - t$, où $t$ est un seuil de contraste. FAST est en $O(1)$ par pixel grâce à un arbre de décision appris qui interrompt le test très tôt après vérification de seulement 3 à 4 pixels en moyenne.

ORB ajoute une **orientation** à chaque point FAST à l'aide de la méthode du **centre de gravité d'intensité**. Les moments d'image du patch $\mathcal{P}$ centré sur le point d'intérêt sont :

$$m_{pq} = \sum_{(x,y) \in \mathcal{P}} x^p \, y^q \, I(x, y)$$

Le centroïde est $\mathbf{C} = (m_{10}/m_{00},\; m_{01}/m_{00})$, et l'orientation du point d'intérêt est l'angle du vecteur allant du centre géométrique vers le centroïde d'intensité :

$$\theta = \text{atan2}(m_{01},\; m_{10})$$

**Descripteur BRIEF.** BRIEF (Binary Robust Independent Elementary Features) code un patch sous la forme d'une chaîne binaire de longueur $n$ (typiquement $n = 256$ bits). Chaque bit résulte d'une comparaison d'intensité entre deux positions du patch $(x_i, y_i)$ et $(x_i', y_i')$ échantillonnées selon un motif appris :

$$\tau(I;\, x, x') = \begin{cases} 1 & \text{si } I(x) < I(x') \\ 0 & \text{sinon} \end{cases}$$

$$\mathbf{f}_n(I) = \sum_{1 \le i \le n} 2^{i-1} \, \tau(I;\, x_i,\, x_i')$$

ORB **fait tourner** les paires d'échantillonnage selon $\theta$ avant de calculer BRIEF, ce qui rend le descripteur invariant à la rotation (rBRIEF). Il applique également une étape gloutonne de décorrélation au motif d'échantillonnage afin de maximiser la variance de chaque bit et de minimiser la corrélation entre bits, ce qui garantit un contenu informatif maximal par bit.

### 2.2 Espace d'échelle et détection multi-échelle

ORB construit une **pyramide d'images** avec `nlevels` niveaux et un facteur d'échelle `scaleFactor` $s > 1$ entre deux niveaux consécutifs. L'image au niveau $k$ est une version sous-échantillonnée de l'original à la résolution $s^{-k}$. Les caractéristiques détectées au niveau $k$ correspondent à des structures de taille spatiale caractéristique $\sim s^k$ pixels dans l'image originale. La pyramide permet la détection de caractéristiques à plusieurs échelles avec un noyau FAST de taille fixe, offrant une **invariance d'échelle** sans modifier le détecteur.

Le budget `nfeatures` est réparti entre les niveaux de la pyramide, avec davantage de caractéristiques allouées aux niveaux les plus fins.

### 2.3 Appariement par distance de Hamming

Comme les descripteurs ORB sont des vecteurs binaires de $\{0,1\}^n$, la mesure naturelle de dissimilarité entre deux descripteurs $\mathbf{d}_1$ et $\mathbf{d}_2$ est la **distance de Hamming** :

$$d_H(\mathbf{d}_1, \mathbf{d}_2) = \|\mathbf{d}_1 \oplus \mathbf{d}_2\|_1 = \text{popcount}(\mathbf{d}_1 \oplus \mathbf{d}_2)$$

où $\oplus$ désigne le XOR bit à bit et `popcount` compte les bits à 1. Pour des descripteurs de 256 bits, cela se ramène à 4 opérations XOR + 4 opérations `popcount` sur une architecture 64 bits, ce qui le rend extrêmement rapide.

Le **brute-force matcher avec cross-check** ne conserve que les paires d'appariement $(i, j)$ telles que la caractéristique $i$ dans l'image gauche ait pour plus proche voisin la caractéristique $j$ dans l'image droite **et** que la caractéristique $j$ ait pour plus proche voisin la caractéristique $i$ dans l'image gauche. Cette contrainte de cohérence mutuelle élimine une grande fraction des faux appariements avant toute vérification géométrique.

---

## 3. Géométrie épipolaire et matrice fondamentale

### 3.1 Contrainte épipolaire

Étant donné un point $\mathbf{p}_L = (u_L, v_L)^\top$ dans l'image gauche, le point correspondant $\mathbf{p}_R$ dans l'image droite ne peut pas se trouver n'importe où dans le plan image ; il doit appartenir à une droite particulière appelée **droite épipolaire**, qui est la projection sur l'image droite du rayon optique passant par $\mathbf{p}_L$. Cette contrainte est encodée par la **matrice fondamentale** $\mathbf{F} \in \mathbb{R}^{3 \times 3}$ :

$$\tilde{\mathbf{p}}_R^\top \, \mathbf{F} \, \tilde{\mathbf{p}}_L = 0$$

où $\tilde{\mathbf{p}} = (u, v, 1)^\top$ désigne les coordonnées homogènes du pixel. La matrice $\mathbf{F}$ est de rang 2 et possède 7 degrés de liberté (9 coefficients, à un facteur d'échelle global près, moins la contrainte de rang 2).

La droite $\ell_R = \mathbf{F} \tilde{\mathbf{p}}_L$ est la droite épipolaire dans l'image droite correspondant à $\mathbf{p}_L$. Symétriquement, $\ell_L = \mathbf{F}^\top \tilde{\mathbf{p}}_R$ est la droite épipolaire dans l'image gauche correspondant à $\mathbf{p}_R$. Après **rectification stéréo**, toutes les droites épipolaires deviennent des lignes horizontales, ce qui réduit la recherche de correspondance d'un problème 2D à un balayage 1D.

### 3.2 Algorithme des 8 points (normalisé)

La matrice fondamentale est estimée à partir des correspondances de points. En développant la contrainte épipolaire avec $\tilde{\mathbf{p}}_L = (x, y, 1)^\top$ et $\tilde{\mathbf{p}}_R = (x', y', 1)^\top$, on obtient une équation linéaire en les 9 coefficients de $\mathbf{F}$ :

$$x'x\,F_{11} + x'y\,F_{12} + x'\,F_{13} + y'x\,F_{21} + y'y\,F_{22} + y'\,F_{23} + x\,F_{31} + y\,F_{32} + F_{33} = 0$$

En empilant $n \geq 8$ telles équations, on obtient le système linéaire homogène $\mathbf{A} \mathbf{f} = \mathbf{0}$, où $\mathbf{f} = \text{vec}(\mathbf{F}) \in \mathbb{R}^9$. La solution est le vecteur singulier à droite de $\mathbf{A}$ associé à sa plus petite valeur singulière, obtenu par SVD. La contrainte de rang 2 est ensuite imposée par une seconde SVD : en écrivant $\mathbf{F} = \mathbf{U} \,\text{diag}(\sigma_1, \sigma_2, \sigma_3)\, \mathbf{V}^\top$ avec $\sigma_1 \geq \sigma_2 \geq \sigma_3$, la matrice de rang 2 la plus proche est :

$$\hat{\mathbf{F}} = \mathbf{U} \,\text{diag}(\sigma_1, \sigma_2, 0)\, \mathbf{V}^\top$$

La **normalisation** (Hartley, 1997) est cruciale pour la stabilité numérique. Avant la résolution, chaque ensemble de points image est transformé indépendamment : translation vers son centroïde, puis mise à l'échelle isotrope de sorte que la distance quadratique moyenne au centroïde soit égale à $\sqrt{2}$. Les transformations de normalisation $\mathbf{T}_L$ et $\mathbf{T}_R$ sont appliquées aux données, la matrice fondamentale est calculée dans ces coordonnées normalisées, puis dénormalisée selon : $\mathbf{F} = \mathbf{T}_R^\top \hat{\mathbf{F}}_{\text{norm}} \mathbf{T}_L$.

### 3.3 RANSAC pour une estimation robuste

Les appariements réels contiennent des **outliers**, c'est-à-dire des paires incorrectement associées à cause de textures répétitives, d'occlusions ou d'ambiguïtés de descripteurs. Les moindres carrés ordinaires sont très sensibles à ces outliers. **RANSAC** (Random Sample Consensus, Fischler & Bolles 1981) traite ce problème selon le schéma itératif suivant :

1. Tirer aléatoirement un **ensemble minimal** de correspondances (8 pour $\mathbf{F}$).
2. Ajuster le modèle, c'est-à-dire estimer $\mathbf{F}$ avec l'algorithme des 8 points normalisé.
3. Compter les **inliers** : toutes les paires $(\mathbf{p}_L, \mathbf{p}_R)$ pour lesquelles la **distance de Sampson** est inférieure à un seuil $\varepsilon$ :

$$d_S(\mathbf{p}_L, \mathbf{p}_R, \mathbf{F}) = \frac{(\tilde{\mathbf{p}}_R^\top \mathbf{F} \tilde{\mathbf{p}}_L)^2}{(\mathbf{F}\tilde{\mathbf{p}}_L)_1^2 + (\mathbf{F}\tilde{\mathbf{p}}_L)_2^2 + (\mathbf{F}^\top\tilde{\mathbf{p}}_R)_1^2 + (\mathbf{F}^\top\tilde{\mathbf{p}}_R)_2^2}$$

La distance de Sampson est une approximation au premier ordre de l'erreur de reprojection, moins coûteuse à calculer que la distance géométrique exacte. Elle mesure dans quelle mesure une paire de points s'écarte de la contrainte épipolaire, normalisée par le gradient local de cette contrainte.

4. Répéter pendant $N$ itérations et conserver l'hypothèse ayant le plus grand nombre d'inliers.
5. Réestimer $\mathbf{F}$ à partir de **tous** les inliers de la meilleure hypothèse à l'aide de l'algorithme des 8 points normalisé.

Le nombre d'itérations $N$ nécessaire pour garantir, avec une probabilité $p$, qu'au moins un échantillon minimal soit exempt d'outliers, pour un taux d'inliers $\rho$, est :

$$N = \frac{\log(1 - p)}{\log(1 - \rho^8)}$$

Pour $p = 0.99$ et un taux d'inliers typique $\rho = 0.5$, on obtient $N \approx 1177$ itérations. Cette application utilise le seuil $\varepsilon = 1.0$ pixel (distance de Sampson) et $p = 0.99$.

---

## 4. Rectification stéréo non calibrée

### 4.1 But de la rectification

L'estimation dense de disparité n'est praticable que sous l'hypothèse que les pixels correspondants appartiennent à la même **ligne de balayage horizontale** dans les deux images. Cela exige que toutes les droites épipolaires soient horizontales, ce qui définit une paire stéréo **rectifiée**. La rectification est mise en oeuvre comme une paire d'homographies planes $\mathbf{H}_L, \mathbf{H}_R \in \mathbb{R}^{3 \times 3}$ qui transforment les deux images de telle sorte que $\ell_R = \mathbf{F}\tilde{\mathbf{p}}_L$ corresponde à une ligne de $v$ constant.

Lorsque les paramètres intrinsèques des caméras sont inconnus, cas **non calibré**, la rectification reste possible à partir de la seule matrice fondamentale $\mathbf{F}$, via l'algorithme de Hartley (1999).

### 4.2 Algorithme de Hartley

Les **épipôles** $\mathbf{e}_L$ et $\mathbf{e}_R$ (projections du centre de chaque caméra dans l'autre image) sont les vecteurs du noyau de $\mathbf{F}^\top$ et de $\mathbf{F}$ respectivement :

$$\mathbf{F}\,\mathbf{e}_L = \mathbf{0}, \qquad \mathbf{F}^\top \mathbf{e}_R = \mathbf{0}$$

La rectification consiste à envoyer les épipôles au point à l'infini suivant l'axe $x$, c'est-à-dire $(1, 0, 0)^\top$ en coordonnées homogènes, ce qui rend toutes les droites épipolaires horizontales. La construction se déroule en trois étapes :

**Étape 1 — Translation.** Les deux images sont translatées de sorte que le centre image soit envoyé à l'origine.

**Étape 2 — Homographie droite $\mathbf{H}_R$.** Une rotation $\mathbf{R}$ est choisie pour envoyer $\mathbf{e}_R$ sur l'axe $x$ (c'est-à-dire vers $(f_e, 0, 1)^\top$ pour une certaine valeur $f_e$). Puis une transformation projective $\mathbf{G}$ envoie ce point fini vers le point idéal $(1, 0, 0)^\top$ :

$$\mathbf{G} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ -1/f_e & 0 & 1 \end{pmatrix}$$

ce qui donne $\mathbf{H}_R = \mathbf{G}\mathbf{R}$ (à un recentrage près).

**Étape 3 — Homographie gauche $\mathbf{H}_L$.** Étant donnée $\mathbf{H}_R$, le but est de trouver $\mathbf{H}_L$ tel que le déplacement vertical entre les points correspondants transformés soit minimisé. L'homographie d'appariement se décompose comme $\mathbf{H}_L = \mathbf{H}_A \mathbf{H}_R \mathbf{M}$, où $\mathbf{M}$ est une transformation dérivée de $\mathbf{F}$ et $\mathbf{H}_A$ est une correction affine. Les coefficients de $\mathbf{H}_A$ sont estimés par **moindres carrés linéaires** sur l'ensemble des correspondances inliers, en minimisant la somme des carrés des résidus verticaux après transformation par $\mathbf{H}_R$ à droite et $\mathbf{H}_L$ à gauche.

Après transformation par $(\mathbf{H}_L, \mathbf{H}_R)$, toute paire de points correspondants vérifie $v_L' \approx v_R'$, ce qui permet une recherche purement horizontale en 1D de la disparité.

### 4.3 Détection automatique de la rectification

Pour décider s'il faut appliquer la rectification, l'application évalue la **médiane des résidus verticaux absolus** sur les correspondances inliers de RANSAC :

$$r_v = \text{median}\bigl(\{|v_{L,i} - v_{R,i}|\}_{i=1}^{N_{\text{in}}}\bigr)$$

Si $r_v \leq \delta$ (défini par l'utilisateur, avec par défaut $\delta = 2$ pixels), la paire est considérée comme déjà rectifiée et aucune homographie n'est appliquée. L'usage de la médiane, plutôt que de la moyenne, rend ce test robuste aux quelques outliers géométriques restants, même parmi les inliers de RANSAC.

---

## 5. Semi-Global Block Matching (SGBM)

### 5.1 Le problème de l'estimation de disparité

Étant données deux images en niveaux de gris rectifiées $I_L(u, v)$ et $I_R(u, v)$ de taille $W \times H$, le but est d'attribuer à chaque pixel de l'image gauche une valeur de disparité $d(u, v) \in \{0, 1, \ldots, D_{\max}\}$ telle que $I_L(u, v) \approx I_R(u - d, v)$.

Il s'agit d'un **problème d'étiquetage dense** dans un espace de recherche $W \times H \times D_{\max}$. Un appariement naïf de type winner-takes-all, où l'on choisit indépendamment pour chaque pixel la valeur de $d$ minimisant un coût local, produit des cartes bruitées comportant de nombreuses erreurs dans les régions peu texturées et près des frontières d'occlusion. Le Semi-Global Matching (SGBM, Hirschmüller 2005, 2008) résout efficacement une version régularisée de ce problème.

### 5.2 Coût de block matching

Pour chaque pixel $(u, v)$ et chaque disparité candidate $d$, le **coût local d'appariement** est calculé sur un bloc carré de côté $b_s$ centré en $(u, v)$ :

$$C_{\text{BM}}(u, v, d) = \sum_{(\Delta u, \Delta v) \in \mathcal{B}} \bigl|I_L(u + \Delta u,\; v + \Delta v) - I_R(u + \Delta u - d,\; v + \Delta v)\bigr|$$

où $\mathcal{B} = \{-\lfloor b_s/2 \rfloor, \ldots, \lfloor b_s/2 \rfloor\}^2$. Il s'agit de la **somme des différences absolues (SAD)** sur la fenêtre d'appariement. La taille de bloc $b_s$ contrôle un compromis fondamental : les petits blocs préservent les frontières de profondeur nettes mais sont sensibles au bruit ; les grands blocs sont plus stables mais sur-lissent les transitions de disparité près des contours des objets.

En pratique, l'implémentation SGBM d'OpenCV préfiltre chaque image par un gradient horizontal tronqué, limité à `preFilterCap = 31`, avant le calcul de SAD, ce qui améliore la robustesse aux différences globales d'illumination entre les deux caméras.

### 5.3 Énergie du semi-global matching

Plutôt que de minimiser $C_{\text{BM}}$ indépendamment en chaque pixel, SGM minimise une énergie globale sur l'ensemble de la carte de disparité $\mathbf{d}$ :

$$E(\mathbf{d}) = \sum_{(u,v)} C_{\text{BM}}(u,v,d(u,v)) \;+\; \sum_{(u,v)} \sum_{(u',v') \in \mathcal{N}(u,v)} P\bigl(d(u,v),\, d(u',v')\bigr)$$

où $\mathcal{N}(u,v)$ désigne l'ensemble des voisins 4- ou 8-connexes, et où la pénalité de paire $P$ est définie par :

$$P(d, d') = \begin{cases} 0 & \text{si } d = d' \\ P_1 & \text{si } |d - d'| = 1 \\ P_2 & \text{si } |d - d'| > 1 \end{cases}$$

$P_1$ pénalise les petits incréments de disparité, ce qui modélise des surfaces à pente douce. $P_2 > P_1$ pénalise les grands sauts, permettant des discontinuités nettes aux frontières des objets tout en les décourageant dans les régions lisses. Dans cette application : $P_1 = 8 b_s^2$ et $P_2 = 32 b_s^2$, de sorte que le rapport $P_2/P_1 = 4$ reste constant quelle que soit la taille du bloc.

La minimisation exacte de $E(\mathbf{d})$ sur une grille 2D est NP-difficile. SGM en propose une approximation efficace selon le schéma suivant.

### 5.4 Programmation dynamique le long de trajectoires

SGM décompose le problème 2D en un ensemble de **programmes dynamiques 1D** le long de $r$ directions, c'est-à-dire des trajectoires rectilignes dans l'image. Pour une trajectoire de direction $\mathbf{r} = (r_u, r_v)$, le **coût de trajectoire** est calculé récursivement d'une extrémité à l'autre :

$$L_\mathbf{r}(u,v,d) = C_{\text{BM}}(u,v,d) + \min \begin{cases} L_\mathbf{r}(u - r_u,\, v - r_v,\; d) \\ L_\mathbf{r}(u - r_u,\, v - r_v,\; d - 1) + P_1 \\ L_\mathbf{r}(u - r_u,\, v - r_v,\; d + 1) + P_1 \\ \displaystyle\min_{d''} L_\mathbf{r}(u - r_u,\, v - r_v,\; d'') + P_2 \end{cases} \;-\; \min_{d''} L_\mathbf{r}(u - r_u,\, v - r_v,\; d'')$$

La soustraction du minimum global à l'étape précédente évite les débordements numériques le long des longues trajectoires sans modifier l'argmin. Le **coût agrégé** s'obtient en sommant sur toutes les directions $\mathbf{r} \in \mathcal{R}$ :

$$S(u,v,d) = \sum_{\mathbf{r} \in \mathcal{R}} L_\mathbf{r}(u,v,d)$$

La disparité finale est alors choisie comme :

$$\hat{d}(u,v) = \arg\min_{d} \, S(u,v,d)$$

Cette application utilise le mode `STEREO_SGBM_MODE_SGBM_3WAY` d'OpenCV, qui agrège les coûts suivant 3 directions et offre un bon compromis vitesse/qualité. La sortie entière brute d'OpenCV est stockée avec **4 bits fractionnaires** (représentation à virgule fixe), si bien que la vraie disparité en pixels s'obtient en divisant par 16.0.

### 5.5 Post-traitement : unicité et filtrage des speckles

**Uniqueness ratio.** Une disparité $\hat{d}(u,v)$ n'est acceptée que si le coût associé à la deuxième meilleure disparité $d^*_2$ (en dehors d'un voisinage $\pm 1$ autour de $\hat{d}$) dépasse le meilleur coût d'au moins une marge relative :

$$S(u,v,d^*_2) > S(u,v,\hat{d}) \cdot \left(1 + \frac{\text{ratio}}{100}\right)$$

Cela rejette les pixels dont le volume de coût est plat, typiquement dans les régions peu texturées (murs, ciel) où l'appariement est ambigu. Les pixels rejetés sont marqués comme invalides.

**Filtre de speckles.** Après l'attribution des disparités, les composantes connexes de la carte de disparité sont analysées. Toute composante vérifiant **les deux conditions** suivantes : (a) son aire est inférieure à `speckleWindowSize` pixels et (b) l'étendue des valeurs de disparité qu'elle contient dépasse `speckleRange`, est considérée comme un artefact de bruit parasite et marquée invalide ($d \to \text{NaN}$). Cela élimine les artefacts isolés sans affecter les véritables discontinuités de profondeur dont les régions connexes sont plus grandes.

---

## 6. Carte de profondeur et reconstruction 3D

### 6.1 De la disparité à la profondeur

À partir de la relation fondamentale de la stéréo $Z = f \cdot b / d$, la **carte de profondeur** est calculée pixel par pixel en appliquant cette formule à la carte de disparité validée. Les pixels pour lesquels $d \leq 0$ ou $d = \text{NaN}$, c'est-à-dire invalides à la suite du filtrage des speckles ou du rejet par unicité, donnent $Z = \text{NaN}$ et sont exclus de tous les traitements suivants.

Pour la visualisation, les cartes de disparité et de profondeur sont rendues avec la **colormap Turbo** après une normalisation robuste par percentiles : la plage d'affichage est tronquée à l'intervalle des percentiles $[2\%, 98\%]$ des valeurs valides, afin d'éviter la saturation due à des outliers extrêmes. La carte de profondeur inverse en plus l'échelle d'intensité de sorte que les objets proches (petit $Z$) apparaissent en couleurs chaudes (rouge/jaune) et les objets éloignés en couleurs froides (bleu/violet), conformément aux conventions usuelles de visualisation de profondeur.

### 6.2 Rétroprojection 3D

Chaque pixel valide $(u, v)$ de disparité $d$ est élevé en coordonnées 3D par inversion simultanée des équations de projection des deux caméras. En approchant le point principal par le centre de l'image $(c_x, c_y) = (W/2, H/2)$ :

$$\begin{pmatrix} X \\ Y \\ Z \end{pmatrix} = \begin{pmatrix} (u - c_x) \cdot b \;/\; d \\ (v - c_y) \cdot b \;/\; d \\ f \cdot b \;/\; d \end{pmatrix}$$

Chaque point 3D hérite de la couleur RGB du pixel correspondant dans l'image gauche. Le nuage de points coloré résultant constitue un échantillonnage discret des surfaces visibles de la scène.

### 6.3 Suppression des valeurs aberrantes et sous-échantillonnage

Malgré le filtrage des speckles, le nuage de points brut peut encore contenir des outliers extrêmes dus à des pixels demi-occultés ou à des effets de bord. Un **écrêtage par percentiles en profondeur** suivant l'axe $Z$ (du 2e au 98e percentile) retire les valeurs les plus extrêmes avant visualisation ou export, assurant ainsi une échelle de visualisation cohérente.

Si le nombre de points 3D valides dépasse la valeur `point_count` définie par l'utilisateur, une stratégie de **sous-échantillonnage uniforme par indice**, consistant à sélectionner des indices régulièrement espacés dans la liste triée des points, réduit le nuage à la taille cible tout en préservant la distribution spatiale globale de la scène. Cela évite le biais de densité que peut introduire un sous-échantillonnage aléatoire près des régions fortement texturées.

---

## 7. Résumé du pipeline

Le pipeline complet de reconstruction, de la paire d'images brute jusqu'au nuage de points 3D, est le suivant :

$$\text{Paire d'images} \;\longrightarrow\; \text{Appariement ORB} \;\longrightarrow\; \text{RANSAC + } \mathbf{F} \;\longrightarrow\; \text{Rectification} \;\longrightarrow\; \text{SGBM} \;\longrightarrow\; \text{Profondeur} \;\longrightarrow\; \text{Rétroprojection 3D}$$

Chaque étape comporte des conditions d'échec explicites qui produisent des erreurs informatives (caractéristiques insuffisantes, trop peu d'inliers RANSAC, dimensions d'image dégénérées pour SGBM), ce qui garantit que l'application se dégrade proprement sur des entrées difficiles au lieu de produire silencieusement des résultats erronés.

---

## 8. Référence des paramètres

| Paramètre | Symbole | Rôle | Plage typique |
|---|---|---|---|
| **Focal Length** (px) | $f$ | Intrinsèque caméra ; met $Z$ à l'échelle linéairement | 500 – 4000 px |
| **Baseline** (m) | $b$ | Écartement des caméras ; met $Z$ à l'échelle linéairement | 0.05 – 1.0 m |
| **Rectification threshold** (px) | $\delta$ | Résidu vertical médian maximal pour éviter la rectification | 1 – 5 px |
| **Number of Disparities** | $D_{\max}$ | Domaine de recherche $[0, D_{\max}]$ ; multiple de 16 | 64 – 512 |
| **Block Size** | $b_s$ | Côté de la fenêtre SAD ; entier impair | 3 – 15 |
| **Uniqueness Ratio** | — | Marge de coût pour l'acceptation (%) | 5 – 15 |
| **Speckle Window Size** | — | Aire minimale de composante connexe à conserver | 0 – 300 |
| **Speckle Range** | — | Variation maximale de disparité dans une composante conservée | 1 – 16 |

**Remarque sur la mise à l'échelle de $Z$.** La profondeur reconstruite est proportionnelle à $f \cdot b$. Si la focale réelle et la baseline réelle sont inconnues, la structure relative de profondeur reste valide, mais les distances métriques seront faussées par un facteur d'échelle constant. Pour les jeux de données de référence comme Middlebury, KITTI ou ETH3D, les valeurs de calibration de référence sont fournies avec les données.

---

## 9. Références

- **Hartley, R. & Zisserman, A.**, *Multiple View Geometry in Computer Vision*, Cambridge University Press, 2e éd., 2004.
- **Hirschmüller, H.**, *Accurate and efficient stereo processing by semi-global matching and mutual information*, CVPR, 2005.
- **Hirschmüller, H.**, *Stereo processing by semiglobal matching and mutual information*, IEEE Transactions on Pattern Analysis and Machine Intelligence, 30(2):328–341, 2008.
- **Rublee, E., Rabaud, V., Konolige, K. & Bradski, G.**, *ORB: An efficient alternative to SIFT or SURF*, ICCV, pp. 2564–2571, 2011.
- **Fischler, M. A. & Bolles, R. C.**, *Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography*, Communications of the ACM, 24(6):381–395, 1981.
- **Hartley, R.**, *In defense of the eight-point algorithm*, IEEE Transactions on Pattern Analysis and Machine Intelligence, 19(6):580–593, 1997.
- **Calonder, M., Lepetit, V., Strecha, C. & Fua, P.**, *BRIEF: Binary Robust Independent Elementary Features*, ECCV, 2010.