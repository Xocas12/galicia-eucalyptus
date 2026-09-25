# Eucalipto en Galicia: informe con datos reais

> **Que é este informe.** Unha estimación con datos de satélite e rexistros públicos do
> efecto das plantacións de eucalipto sobre o bosque autóctono e os incendios en Galicia.
> Os mapas de especies adestráronse con etiquetas de OpenStreetMap, non co Mapa Forestal de
> España nin co Inventario Forestal Nacional, que non eran accesibles desde este contorno.
> **O mapa de eucalipto só está validado no norte (cadro de 100 km que abrangue A Coruña, Ferrol e Ortegal),
> onde están o 91 % das etiquetas de eucalipto.** Nunha validación que deixa fóra
> cadros enteiros de 100 km, o F1 do eucalipto cae a 0,0027
> (2024) e 0,0098 (2017): fóra desa rexión o mapa non
> distingue o eucalipto de forma fiable, e iso afecta a todas as estimacións que usan a
> fracción de eucalipto.
> Os efectos causais dependen de supostos que se explican na sección 7. O efecto sobre a auga
> **non se puido estimar** con datos reais (sección 6).

## Resumo

- **Superficie de eucalipto (mapa).** 144 mil ha en 2024 e 193 mil ha en 2017 (reconto de píxeles; 167 mil ha en 2024 sumando probabilidades). **Estas cifras non están validadas** co inventario oficial e quedan claramente por debaixo das superficies de eucalipto publicadas para Galicia, polo que é probable que o mapa infraestime o eucalipto (sección 2). Non se deben citar como superficie oficial.
- **Substitución de bosque autóctono.** Entre 2017 e 2024, 5 729 ha pasaron de frondosas autóctonas a eucalipto en píxeles clasificados con fiabilidade nos dous anos; 974,4 ha diso coinciden ademais cunha perda de cuberta arbórea (Hansen) ou cun incendio (EFFIS). Esta última é a cifra máis prudente; a diferenza entre mapas tende a sobreestimar o cambio.
- **Incendios, 2018–2023.** Mantendo constantes o relevo, a localización, a presión humana, a meteoroloxía e as demais cubertas, **non se detecta un efecto do eucalipto** sobre a probabilidade anual de queima distinguible de cero, fronte a agricultura e outros usos: por cada 10 puntos de eucalipto, -0,2 puntos porcentuais (IC 95 %: -0,53 a 0,14), cunha taxa base de 1,2 % ao ano. O mato si aumenta o risco: 0,33 puntos porcentuais por cada 10 puntos de mato (IC 95 %: 0,11 a 0,56). Na única rexión onde o mapa está validado, o efecto é 0,068 puntos porcentuais por cada 10 puntos de eucalipto (IC 95 %: -0,33 a 0,47; 24 anos-cela queimados), tampouco distinguible de cero. Dado que o mapa de eucalipto non é fiable fóra do norte, a ausencia de efecto non demostra que o eucalipto non afecte aos incendios: o dato non ten potencia para decidilo.
- **Severidade.** Entre as celas queimadas, o efecto do eucalipto sobre a clase de severidade EFFIS é -0,05 por unidade de fracción (EE 0,71; non distinguible de cero; n = 2 146).
- **Do lume á plantación.** Efecto da fracción queimada en 2018–2021 sobre a conversión bruta a eucalipto en 2024: 0,0013 (EE 0,002), non distinguible de cero.
- **Proxeccións a 2040.** Restaurar o 25 % do eucalipto cambia a superficie queimada media en 430,1 ha/ano se se fai nas celas prioritarias (banda 5–95 %: -160,9 a 1 046) e en 337,5 ha/ano se se fai ao chou (banda -54,43 a 731,8). Como os efectos das cubertas sobre o lume non son distinguibles de cero, as proxeccións non permiten afirmar que restaurar reduza os incendios; tampouco que os aumente.

## 1. Datos empregados

| fonte | uso | período |
|---|---|---|
| Sentinel-2 L2A (Copernicus, arquivo COG de AWS) | mapas de especies (NDVI, NDMI, NBR mensuais, 40 m) | nov. 2016–set. 2017 (o arquivo L2A comeza en nov. 2016) e out. 2023–set. 2024 |
| OpenStreetMap (vía Overture Maps) | etiquetas de adestramento (tipo de folla, xénero, mato, prados) | 2026 |
| ESA WorldCover | máscara de arboredo e clases non forestais | 2021 |
| Hansen Global Forest Change v1.12 | perda de cuberta arbórea anual, cuberta en 2000 | 2001–2024 |
| EFFIS (severidade de queimados) | área queimada e severidade | 2018–2023 |
| Copernicus DEM (90 m) | altitude e pendente | estático |
| NOAA GHCN-Daily (6 estacións galegas) | anomalías de temperatura e choiva estivais | 2001–2025 |
| Overture Maps, edificacións | presión humana (ignicións) | 2026 |
| Natural Earth | límite de Galicia (catro provincias) | estático |

Panel: 29 565 celas de 1 km e 177 390 anos-cela (2018–2023), dos cales
2 146 rexistraron queimados.

![mapas](mapas.png)

## 2. Mapas de especies

Clasificador de potenciación de gradiente sobre trazos fenolóxicos (media, amplitude e fase anual
de NDVI, NDMI e NBR). Exactitude en validación cruzada por bloques espaciais de 20 km:
**0,79** (2017) e **0,82**
(2024). Kappa 0,748 e 0,784. A exactitude mide o acordo coas
etiquetas de OpenStreetMap, que non son unha mostra aleatoria.

Validación por rexións (deixando fóra cadros enteiros de 100 km), F1 por clase:

| clase | 2017 | 2024 |
|---|---|---|
| eucalipto | 0,00983 | 0,00273 |
| piñeiro | 0,665 | 0,69 |
| frondosas autóctonas | 0,54 | 0,546 |
| mato | 0,663 | 0,709 |
| agricultura | 0,73 | 0,805 |
| outros | 0,881 | 0,897 |

O eucalipto non se transfire a rexións sen etiquetas: o 91 % dos píxeles de adestramento de
eucalipto están nun só cadro de 100 km.

Superficies. A columna «superficie estimada» corrixe o mapa invertindo a matriz de confusión
das etiquetas de OpenStreetMap. Esa corrección só é fiable se as etiquetas son puras: se algúns
polígonos etiquetados como frondosas autóctonas conteñen eucalipto, a inversión resta
eucalipto de máis. Aquí reduce o eucalipto a unha fracción do mapa e dá un descenso entre 2017
e 2024 que o propio mapa non mostra, sinal de que as etiquetas non son abondo puras. **Tómese
como unha comprobación fráxil, non como estimación.** A exactitude do usuario do eucalipto
(a probabilidade de que un píxel mapeado como eucalipto o sexa) é a cifra máis débil do mapa.

2024:

| clase | superficie no mapa (ha) | superficie estimada (ha) | IC 95 % (± ha) | superficie por probabilidades (ha) | exactitude do usuario | exactitude do produtor |
|---|---|---|---|---|---|---|
| eucalipto | 144 288 | 57 505 | 4 283 | 166 654 | 0,31324 | 0,78597 |
| piñeiro | 688 023 | 657 456 | 6 153 | 712 764 | 0,78204 | 0,8184 |
| frondosas autóctonas | 731 836 | 841 271 | 7 019 | 721 058 | 0,87196 | 0,75853 |
| mato | 405 139 | 363 005 | 5 041 | 422 422 | 0,70928 | 0,7916 |
| agricultura | 745 070 | 843 553 | 4 899 | 735 530 | 0,9382 | 0,82867 |
| outros | 243 352 | 194 919 | 2 455 | 278 471 | 0,75078 | 0,93733 |

2017:

| clase | superficie no mapa (ha) | superficie estimada (ha) | IC 95 % (± ha) | superficie por probabilidades (ha) | exactitude do usuario | exactitude do produtor |
|---|---|---|---|---|---|---|
| eucalipto | 192 761 | 121 807 | 4 526 | 222 085 | 0,47007 | 0,7439 |
| piñeiro | 724 643 | 697 682 | 6 146 | 721 009 | 0,78487 | 0,8152 |
| frondosas autóctonas | 664 390 | 734 884 | 7 628 | 678 017 | 0,81465 | 0,7365 |
| mato | 432 786 | 391 449 | 6 017 | 442 086 | 0,67662 | 0,74807 |
| agricultura | 710 629 | 849 646 | 5 825 | 700 548 | 0,91752 | 0,7674 |
| outros | 232 500 | 162 240 | 2 978 | 273 154 | 0,64984 | 0,93127 |

## 3. Perda de bosque autóctono

Transicións cara ao eucalipto, 2017–2024 (píxeles de 40 m):

| de | a | superficie (ha) | superficie, píxeles fiables (ha) |
|---|---|---|---|
| piñeiro | eucalipto | 19 705 | 2 374 |
| frondosas autóctonas | eucalipto | 32 930 | 5 729 |
| mato | eucalipto | 9 250 | 1 151 |
| agricultura | eucalipto | 9 953 | 1 785 |
| outros | eucalipto | 1 182 | 137,92 |

A diferenza entre dous mapas acumula os erros de ambos e sobreestima o cambio (na validación
sintética, ao redor do dobre). Por iso o informe dá tres cifras para autóctonas → eucalipto:
todos os píxeles, 32 930 ha; píxeles fiables, 5 729 ha;
e píxeles fiables con perda arbórea ou incendio que o corroboren,
974 ha. A última é a máis prudente.

Atribución da perda de cuberta arbórea 2018–2024 (Hansen) a 40 m:

| causa | atribuída | proporción atribuída |
|---|---|---|
| incendio | 42 724 | 0,1717 |
| corta de rotación | 83 484 | 0,3354 |
| conversión | 8 535 | 0,03429 |
| perda de frondosas sen conversión | 34 126 | 0,1371 |
| sen atribuír | 80 028 | 0,3215 |

## 4. Incendios

| efecto | método | estimación | EE | n |
|---|---|---|---|---|
| eucalipto → P(queima) | MCO | -0,03332 | 0,007956 | 177 390 |
| eucalipto → P(queima) | DML | -0,01976 | 0,01706 | 177 390 |
| eucalipto → fracción queimada | MCO | -0,01494 | 0,004221 | 177 390 |
| eucalipto → fracción queimada | DML | -0,00332 | 0,007234 | 177 390 |
| eucalipto → severidade (clase EFFIS) | MCO | -1,464 | 0,7542 | 2 146 |
| eucalipto → severidade (clase EFFIS) | DML | -0,05027 | 0,7125 | 2 146 |
| queimado 2018-2021 → ganancia de eucalipto | MCO | -0,02379 | 0,005811 | 28 983 |
| queimado 2018-2021 → ganancia de eucalipto | DML | 0,001299 | 0,001983 | 28 983 |

![efectos](efectos.png)

Efecto de cada cuberta sobre a probabilidade anual de queima, fronte a agricultura e outros usos:

| cuberta | dP(queima)/dfracción | EE |
|---|---|---|
| eucalipto | -0,01976 | 0,01706 |
| piñeiro | 0,004733 | 0,008907 |
| frondosas autóctonas | -0,003166 | 0,01161 |
| mato | 0,03344 | 0,01152 |

Onde é maior o efecto do eucalipto:

| grupo | estimación | EE | n |
|---|---|---|---|
| costa | -0,01672 | 0,02122 | 59 208 |
| transición | -0,04074 | 0,02687 | 59 100 |
| interior | 0,01131 | 0,04052 | 59 082 |

| grupo | estimación | EE | n |
|---|---|---|---|
| FWI baixo | -0,03921 | 0,01732 | 59 130 |
| FWI medio | 0,01519 | 0,01558 | 59 130 |
| FWI alto | -0,04419 | 0,03793 | 59 130 |

![grupos](grupos.png)

Sensibilidade á confusión non observada:

| efecto | estimación | VR da estimación | VR do IC | nesgo máx. (R² = 0,02) | nesgo máx. (R² = 0,05) |
|---|---|---|---|---|---|
| eucalipto → P(queima) | -0,01976 | 0,007802 | 0 | 0,05097 | 0,1294 |
| eucalipto → fracción queimada | -0,00332 | 0,002477 | 0 | 0,02705 | 0,06868 |
| eucalipto → severidade (clase EFFIS) | -0,05027 | 0,003188 | 0 | 0,3181 | 0,8076 |

Erro estándar do efecto sobre a aparición de incendios segundo o tamaño do bloque:

| bloque (km) | EE |
|---|---|
| 5 | 0,009481 |
| 10 | 0,0128 |
| 20 | 0,01486 |
| 40 | 0,01671 |
| 80 | 0,02402 |

Modelo de susceptibilidade: AUC en validación cruzada espacial 0,872.

## 5. Proxeccións ano a ano, 2025–2040

Motor dinámico: cada ano sortéase o lume segundo a susceptibilidade de cada cela e os efectos
causais estimados de cada cuberta; o lume converte parte das frondosas e dos piñeirais en mato;
e a plantación de eucalipto segue a taxa de conversión bruta observada en 2017–2024 entre
píxeles fiables (0,22 % ao ano da
superficie sen eucalipto), reforzada polos incendios recentes. O motor non inclúe perdas de
eucalipto agás a restauración, así que o crecemento no escenario tendencial é un límite superior. O modelo validouse na paisaxe sintética (1 km), onde sobreestimou uns 40 % os beneficios
da restauración e moito máis os do límite, porque sobreestima a plantación de referencia. As bandas son os percentís 5 e 95 de 40 simulacións que combinan a
variabilidade meteorolóxica e a incerteza dos efectos.

| escenario | Δ eucalipto (ha) | Δ frondosas autóctonas (ha) | Δ queimado medio (ha/ano) | percentil 5 | percentil 95 |
|---|---|---|---|---|---|
| Límite / moratoria | -56 879 | 21 628 | 273,9 | 42,46 | 511,6 |
| Restauración dirixida | -96 223 | 60 795 | 430,1 | -160,9 | 1 046 |
| Restauración aleatoria | -96 217 | 60 898 | 337,5 | -54,43 | 731,8 |

![proxeccións](proxeccions.png)

Sensibilidade: se a plantación futura fose a metade da observada en 2017–2024 (por exemplo,
porque se manteñen as restricións a novas plantacións):

| escenario | Δ eucalipto (ha) | Δ frondosas autóctonas (ha) | Δ queimado medio (ha/ano) | percentil 5 | percentil 95 |
|---|---|---|---|---|---|
| Límite / moratoria | -34 039 | 12 973 | 229,8 | 38,14 | 439 |

## 6. Auga

Non se estimou. Os datos de caudal (Augas de Galicia, anuario de aforos do CEDEX) non eran
accesibles desde este contorno, e ningunha fonte alcanzable medía a escorrentía. O código para
os paneis de concas e a curva de Budyko está listo e validado con datos sintéticos; só precisa
os caudais diarios das estacións.

## 7. Limitacións

- **Etiquetas.** As etiquetas de especie proceden de OpenStreetMap: 30 000
  píxeles de adestramento de eucalipto en 2024, a clase con menos exemplos. Supúxose que o
  bosque frondoso perennifolio de Galicia é eucalipto; as aciñeiras e sobreiras quedarían mal
  clasificadas. Hai que validar os mapas co Mapa Forestal de España ou co IFN4.
- **Erro do mapa.** O erro do mapa atenúa os efectos cara a cero. Non se aplicou SIMEX porque
  non hai unha mostra de referencia independente para medir a varianza do erro.
- **Incendios.** EFFIS rexistra sobre todo os incendios grandes; os pequenos quedan fóra. Só hai
  seis anos (2018–2023) e 2022 domina o total.
- **Causalidade.** Os efectos son causais só se non queda confusión relevante sen medir (por
  exemplo, a intencionalidade dos lumes ou a propiedade das terras). A táboa de sensibilidade
  indica canto tería que pesar ese factor.
- **Meteoroloxía.** O índice meteorolóxico procede de seis estacións; non é o FWI oficial.

## Siglas

| sigla | significado |
|---|---|
| AUC | área baixo a curva ROC |
| DML | aprendizaxe automática dobre (estimación causal con axustes cruzados) |
| EE | erro estándar |
| EFFIS | Sistema Europeo de Información sobre Incendios Forestais |
| FWI | índice meteorolóxico de perigo de incendio |
| IC | intervalo de confianza |
| MCO | mínimos cadrados ordinarios (estimación inxenua) |
| VR | valor de robustez |
