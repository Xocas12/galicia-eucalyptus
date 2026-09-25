# Eucalipto en Galicia: informe con datos reais

> **Que é este informe.** Unha estimación con datos de satélite e rexistros públicos do
> efecto das plantacións de eucalipto sobre o bosque autóctono e os incendios en Galicia.
> Os mapas de especies adestráronse con etiquetas de OpenStreetMap, non co Mapa Forestal de
> España nin co Inventario Forestal Nacional, que non eran accesibles desde este contorno.
> **Como se validou o mapa de eucalipto.** Case todas as etiquetas de eucalipto de
> OpenStreetMap están nun cadro de 100 km do norte (A Coruña, Ferrol, Ortegal), e moitas das
> de fóra teñen un comportamento invernal de frondosa caducifolia, é dicir, están mal
> etiquetadas. Por iso as etiquetas límpanse segundo o comportamento invernal (o eucalipto é
> perennifolio) e engádense pseudoetiquetas de eucalipto en toda Galicia: arboredo verde e
> húmido no inverno en contornas con cortas a matarrasa 2001–2016. Proba de transferencia:
> adestrando sen o cadro do norte e avaliando nas súas etiquetas de OpenStreetMap, o F1 do
> eucalipto é 0,72 en 2024 (precisión
> 0,69, sensibilidade
> 0,75) e
> 0,65 en 2017. Sen esta corrección era
> practicamente cero. Segue sen haber unha mostra de referencia independente fóra do norte.
> Os efectos causais dependen de supostos que se explican na sección 7. O efecto sobre a auga
> **non se puido estimar** con datos reais (sección 6).

## Resumo

- **Superficie de eucalipto (mapa).** 440 mil ha en 2024 e 489 mil ha en 2017 (reconto de píxeles; 447 mil ha en 2024 sumando probabilidades). **Estas cifras non están validadas** co inventario oficial (IFN, Mapa Forestal de España): compárense con el antes de citalas (sección 2).
- **Substitución de bosque autóctono.** Entre 2017 e 2024, 563,2 ha pasaron de frondosas autóctonas a eucalipto en píxeles clasificados con fiabilidade nos dous anos; 484 ha diso coinciden ademais cunha perda de cuberta arbórea (Hansen) ou cun incendio (EFFIS). Esta última é a cifra máis prudente; a diferenza entre mapas tende a sobreestimar o cambio.
- **Incendios, 2018–2023.** Mantendo constantes o relevo, a localización, a presión humana, a meteoroloxía e as demais cubertas, un aumento de 10 puntos na fracción de eucalipto cambia a probabilidade anual de queima en -0,27 puntos porcentuais (IC 95 %: -0,46 a -0,076). O mato non mostra un efecto distinguible de cero. O valor de robustez é 0,018: un factor de confusión non medido con ese R² parcial co tratamento e co resultado anularía a estimación. No cadro do norte, onde están as etiquetas de OpenStreetMap, o efecto é -0,34 puntos porcentuais por cada 10 puntos de eucalipto (IC 95 %: -0,54 a -0,14; 24 anos-cela queimados). As versións do mapa coinciden no signo (2017 retrodatado: -0,27, 2017 independente: -0,15, 2024 (posterior aos lumes): -0,42), pero non todas son distinguibles de cero: o resultado é sensible ao mapa.
- **Eucalipto fronte a frondosas autóctonas.** O efecto anterior compárase coa agricultura e outros usos, que son os que máis arden. Fronte ás frondosas autóctonas, que son as que menos arden, 10 puntos de eucalipto no canto de frondosas cambian a probabilidade anual de queima en 0,39 puntos porcentuais (IC 95 % aproximado: -0,018 a 0,79; aproximado porque combina dúas estimacións separadas). Este é o contraste que importa para a restauración.
- **Severidade.** Entre as celas queimadas, o efecto do eucalipto sobre a clase de severidade EFFIS é -0,24 por unidade de fracción (EE 0,28; non distinguible de cero; n = 2 146).
- **Do lume á plantación.** Efecto da fracción queimada en 2018–2021 sobre a conversión bruta a eucalipto en 2024: 0,0063 (EE 0,0088), non distinguible de cero.
- **Proxeccións a 2040.** Restaurar o 25 % do eucalipto cambia a superficie queimada media en -2 499 ha/ano se se fai nas celas prioritarias (banda 5–95 %: -3 577 a -866,9) e en -714,8 ha/ano se se fai ao chou (banda -1 077 a -244,2). Ao menos unha banda exclúe o cero, pero as proxeccións herdan a sensibilidade ao mapa descrita arriba.

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
**0,875** (2017) e **0,875**
(2024). Kappa 0,845 e 0,845. A exactitude mide o acordo coas
etiquetas de OpenStreetMap, que non son unha mostra aleatoria.

Proba de transferencia (adestramento sen o cadro do norte, avaliación nas súas etiquetas de
OpenStreetMap), F1 por clase:

| clase | 2017 | 2024 |
|---|---|---|
| eucalipto | 0,652 | 0,721 |
| piñeiro | 0,0518 | 0,0895 |
| frondosas autóctonas | 0,314 | 0,461 |
| mato | 0,037 | 0,166 |
| agricultura | 0,28 | 0,287 |
| outros | 0,966 | 0,969 |

Pseudoetiquetas de eucalipto engadidas: 39 445 píxeles
(o mesmo modelo clasifica os dous anos).

O mapa de 2017 retrodátase desde o de 2024: as imaxes de 2017 normalízanse radiometricamente
contra as de 2024 e clasifícanse co mesmo modelo, pero nos píxeles sen perturbación entre os dous
anos (sen perda arbórea de Hansen nin queimado de EFFIS) mantense a clase de 2024. Así, o
ruído do clasificador non crea cambios falsos, e un cambio real precisa de evidencia. Retrodatouse o
90 % dos píxeles.

Superficies. A columna «superficie estimada» corrixe o mapa invertindo a matriz de confusión
da validación cruzada. Esa corrección só é fiable se as etiquetas son puras e representativas;
as de OpenStreetMap non o son, así que **tómese como unha comprobación, non como estimación**.
Se se afasta moito da superficie do mapa, a diferenza indica ruído nas etiquetas máis ca un
erro do mapa.

2024:

| clase | superficie no mapa (ha) | superficie estimada (ha) | IC 95 % (± ha) | superficie por probabilidades (ha) | exactitude do usuario | exactitude do produtor |
|---|---|---|---|---|---|---|
| eucalipto | 439 558 | 402 700 | 2 312 | 447 317 | 0,87601 | 0,95619 |
| piñeiro | 625 464 | 643 905 | 5 557 | 636 351 | 0,81643 | 0,79305 |
| frondosas autóctonas | 522 367 | 509 358 | 3 967 | 528 704 | 0,85269 | 0,87446 |
| mato | 389 633 | 364 617 | 4 832 | 405 262 | 0,72006 | 0,76947 |
| agricultura | 736 967 | 850 757 | 5 236 | 745 316 | 0,94326 | 0,8171 |
| outros | 243 718 | 186 370 | 2 777 | 273 951 | 0,71746 | 0,93823 |

2017:

| clase | superficie no mapa (ha) | superficie estimada (ha) | IC 95 % (± ha) | superficie por probabilidades (ha) | exactitude do usuario | exactitude do produtor |
|---|---|---|---|---|---|---|
| eucalipto | 488 611 | 455 084 | 2 322 | 497 303 | 0,89058 | 0,95619 |
| piñeiro | 608 352 | 624 847 | 5 467 | 619 094 | 0,81455 | 0,79305 |
| frondosas autóctonas | 504 485 | 490 814 | 3 839 | 510 368 | 0,85077 | 0,87446 |
| mato | 370 649 | 340 669 | 4 751 | 384 981 | 0,70723 | 0,76947 |
| agricultura | 744 857 | 862 109 | 5 258 | 753 734 | 0,94572 | 0,8171 |
| outros | 240 753 | 184 184 | 2 744 | 271 420 | 0,71778 | 0,93823 |

## 3. Perda de bosque autóctono

Transicións cara ao eucalipto, 2017–2024 (píxeles de 40 m):

| de | a | superficie (ha) | superficie, píxeles fiables (ha) |
|---|---|---|---|
| piñeiro | eucalipto | 7 704 | 3 992 |
| frondosas autóctonas | eucalipto | 1 548 | 563,2 |
| mato | eucalipto | 2 910 | 1 123 |
| agricultura | eucalipto | 7 038 | 3 816 |
| outros | eucalipto | 277,92 | 79,68 |

A diferenza entre dous mapas acumula os erros de ambos e sobreestima o cambio (na validación
sintética, ao redor do dobre). Por iso o informe dá tres cifras para autóctonas → eucalipto:
todos os píxeles, 1 548 ha; píxeles fiables, 563 ha;
e píxeles fiables con perda arbórea ou incendio que o corroboren,
484 ha. A última é a máis prudente.

Atribución da perda de cuberta arbórea 2018–2024 (Hansen) a 40 m:

| causa | atribuída | proporción atribuída |
|---|---|---|
| incendio | 42 724 | 0,1717 |
| corta de rotación | 112 951 | 0,4538 |
| conversión | 6 421 | 0,0258 |
| perda de frondosas sen conversión | 8 613 | 0,03461 |
| sen atribuír | 78 187 | 0,3141 |

## 4. Incendios

| efecto | método | estimación | EE | n |
|---|---|---|---|---|
| eucalipto → P(queima) | MCO | -0,01557 | 0,004457 | 177 390 |
| eucalipto → P(queima) | DML | -0,02679 | 0,009813 | 177 390 |
| eucalipto → fracción queimada | MCO | -0,005055 | 0,002113 | 177 390 |
| eucalipto → fracción queimada | DML | -0,01644 | 0,007109 | 177 390 |
| eucalipto → severidade (clase EFFIS) | MCO | -0,4024 | 0,1758 | 2 146 |
| eucalipto → severidade (clase EFFIS) | DML | -0,2377 | 0,2825 | 2 146 |
| queimado 2018-2021 → ganancia de eucalipto | MCO | -0,00227 | 0,01078 | 29 320 |
| queimado 2018-2021 → ganancia de eucalipto | DML | 0,006307 | 0,00882 | 29 320 |

![efectos](efectos.png)

Efecto de cada cuberta sobre a probabilidade anual de queima, fronte a agricultura e outros usos:

| cuberta | dP(queima)/dfracción | EE |
|---|---|---|
| eucalipto | -0,02679 | 0,009813 |
| piñeiro | -0,0367 | 0,01654 |
| frondosas autóctonas | -0,06539 | 0,01812 |
| mato | -0,01624 | 0,01342 |

Onde é maior o efecto do eucalipto:

| grupo | estimación | EE | n |
|---|---|---|---|
| costa | -0,01871 | 0,008733 | 59 208 |
| transición | -0,05103 | 0,02186 | 59 100 |
| interior | -0,0007126 | 0,01633 | 59 082 |

| grupo | estimación | EE | n |
|---|---|---|---|
| FWI baixo | -0,009122 | 0,009367 | 59 130 |
| FWI medio | 0,01047 | 0,006363 | 59 130 |
| FWI alto | -0,07721 | 0,02989 | 59 130 |

![grupos](grupos.png)

Sensibilidade ao mapa de eucalipto (efecto sobre a probabilidade anual de queima por unidade
de fracción):

| mapa | estimación | EE | IC 95 % inferior | IC 95 % superior |
|---|---|---|---|---|
| 2017 retrodatado | -0,02679 | 0,009813 | -0,04602 | -0,007552 |
| 2017 independente | -0,0148 | 0,0076 | -0,0297 | 9,827e-05 |
| 2024 (posterior aos lumes) | -0,04163 | 0,012 | -0,06515 | -0,0181 |

Sensibilidade á confusión non observada:

| efecto | estimación | VR da estimación | VR do IC | nesgo máx. (R² = 0,02) | nesgo máx. (R² = 0,05) |
|---|---|---|---|---|---|
| eucalipto → P(queima) | -0,02679 | 0,01763 | 0,005002 | 0,03043 | 0,07726 |
| eucalipto → fracción queimada | -0,01644 | 0,02165 | 0,003332 | 0,01517 | 0,03853 |
| eucalipto → severidade (clase EFFIS) | -0,2377 | 0,02479 | 0 | 0,1913 | 0,4857 |

Erro estándar do efecto sobre a aparición de incendios segundo o tamaño do bloque:

| bloque (km) | EE |
|---|---|
| 5 | 0,006694 |
| 10 | 0,00835 |
| 20 | 0,01057 |
| 40 | 0,01288 |
| 80 | 0,01438 |

Modelo de susceptibilidade: AUC en validación cruzada espacial 0,885.

## 5. Proxeccións ano a ano, 2025–2040

Motor dinámico: cada ano sortéase o lume segundo a susceptibilidade de cada cela e os efectos
causais estimados de cada cuberta; o lume converte parte das frondosas e dos piñeirais en mato;
e a plantación de eucalipto segue a taxa de conversión bruta observada en 2017–2024 entre
píxeles fiables (0,16 % ao ano da
superficie sen eucalipto), reforzada polos incendios recentes. O motor non inclúe perdas de
eucalipto agás a restauración, así que o crecemento no escenario tendencial é un límite superior. O modelo validouse na paisaxe sintética (1 km), onde sobreestimou uns 40 % os beneficios
da restauración e moito máis os do límite, porque sobreestima a plantación de referencia. As bandas son os percentís 5 e 95 de 40 simulacións que combinan a
variabilidade meteorolóxica e a incerteza dos efectos.

| escenario | Δ eucalipto (ha) | Δ frondosas autóctonas (ha) | Δ queimado medio (ha/ano) | percentil 5 | percentil 95 |
|---|---|---|---|---|---|
| Límite / moratoria | -24 918 | 1 663 | 22,85 | -33,92 | 66,07 |
| Restauración dirixida | -130 892 | 107 637 | -2 499 | -3 577 | -866,9 |
| Restauración aleatoria | -130 863 | 107 608 | -714,8 | -1 077 | -244,2 |

![proxeccións](proxeccions.png)

Sensibilidade: se a plantación futura fose a metade da observada en 2017–2024 (por exemplo,
porque se manteñen as restricións a novas plantacións):

| escenario | Δ eucalipto (ha) | Δ frondosas autóctonas (ha) | Δ queimado medio (ha/ano) | percentil 5 | percentil 95 |
|---|---|---|---|---|---|
| Límite / moratoria | -13 477 | 816,6 | 12,44 | -17,35 | 34,82 |

## 6. Auga

Non se estimou. Os datos de caudal (Augas de Galicia, anuario de aforos do CEDEX) non eran
accesibles desde este contorno, e ningunha fonte alcanzable medía a escorrentía. O código para
os paneis de concas e a curva de Budyko está listo e validado con datos sintéticos; só precisa
os caudais diarios das estacións.

## 7. Limitacións

- **Etiquetas.** As etiquetas de especie proceden de OpenStreetMap: 65 083
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
