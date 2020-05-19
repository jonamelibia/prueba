##########################################################################################
##### Evaluación de clasificadores

# Podemos evaluar la efectividad de un modelo alterando diferentes parámetros y elegir el mejor.
library(recommenderlab)
library(ggplot2)
data(MovieLense)
MovieLense

ratings_movies <- MovieLense[rowCounts(MovieLense) > 50,
                             colCounts(MovieLense) > 100]

ratings_movies

### 1a opción. Splittear los datos

# El test set se dividiróá en 2. Un grupo de individuos con los que probaremos el accuracy
# y otro grupo de individuos que usaremos para predecir. El número de items para generar recomendaciones
# debe ser menor que el mínimo de artículos adquiridos por cada usuario, para que
# no tengamos usuarios sin items para probar los modelos

min(rowCounts(ratings_movies)) # si el mínimo es 18, nos quedaremos al menos con 17

eval_sets <- evaluationScheme(data = ratings_movies, method = "split", train = 0.8,
                              given = 15, goodRating = 3)
# train : train set
# known : test set para construir las recomendaciones
# unknown : test set para evaluar las recomendaciones

nrow(getData(eval_sets, "train"))/nrow(ratings_movies)
nrow(getData(eval_sets, "train"))
nrow(getData(eval_sets, "known"))/nrow(ratings_movies)
nrow(getData(eval_sets, "known"))
nrow(getData(eval_sets, "unknown")) # tienen el mismo num. de usuarios y son el 20%
unique(rowCounts(getData(eval_sets, "known"))) # 15 artículos por usuario

### 2a opción. Bootstrapping

eval_sets <- evaluationScheme(data = ratings_movies, method = "bootstrap", train = 0.8,
                              given = 15, goodRating = 3)

nrow(getData(eval_sets, "train"))/nrow(ratings_movies)
nrow(getData(eval_sets, "train"))
nrow(getData(eval_sets, "known"))/nrow(ratings_movies) # % de usuarios de test
nrow(getData(eval_sets, "known"))
nrow(getData(eval_sets, "unknown"))
length(unique(eval_sets@runsTrain[[1]])) # pero ¿Cuántos usuarios únicos hay en trainú
length(unique(eval_sets@runsTrain[[1]]))/nrow(ratings_movies) # % de usuarios únicos de train

eval_sets@runsTrain[[1]]
eval_sets@knownData
as(eval_sets@knownData, "list")[60] # elementos que han ido a la parte test1 o known
as(eval_sets@unknownData, "list")[60] # elementos que han ido a la parte test2 o unknown
as(eval_sets@data, "list")[60]

# ¿Cómo sé cuántas veces está repetido un usuario en train?
table_train <- table(eval_sets@runsTrain[[1]])
n_repetitions <- factor(as.vector(table_train))
qplot(n_repetitions) + ggtitle("Número de repeticiones en el train set") +
  xlab("Número de repeticiones") + ylab("Número de usuarios")

### 3a opción. K-fold validation (uso preferencial)

eval_sets <- evaluationScheme(data = ratings_movies, method = "cross-validation", train = 0.8,
                              given = 15, goodRating = 3, k = 4)

size_sets <- sapply(eval_sets@runsTrain,length)
size_sets

##############################################################################################
### Evaluación de los modelos (Ratings)

eval_reccomender <- Recommender(getData(eval_sets,"train"),
                                "IBCF", 
                                param = NULL)
eval_reccomender

eval_prediction <- predict(object = eval_reccomender, 
                           newdata = getData(eval_sets, "known"),
                           n = 10,
                           type = "ratings")

eval_prediction@data # muestra la matriz de puntuaciones predichas
eval_prediction@normalize # no se ha normalizado

eval_accuracy <- calcPredictionAccuracy(eval_prediction,
                                        getData(eval_sets, "unknown"),
                                        byUser = TRUE)
head(eval_accuracy)

### Evaluación de los modelos (Recomendaciones)

results <- evaluate(eval_sets, "IBCF", n=seq(10,100,10))
class(results)
head(getConfusionMatrix(results)[[1]]) # nos muestra una lista de MC. Cada elemento es un split del k-fold
  # TP: artículos recomendados que han sido comprados
  # FP: artículos recomendados que no han sido comprados
  # FN: artículos no recomendados que han sido comprados
  # TN: artículos no recomendados que no han sido comprados
  # Un modelo perfecto o sobreajustado sílo tendría TP y TN.
avg(results) # nos representa el promedio de todos los splits

# Curva ROC
  # TPR: True positive rate
  # % de artículos comprados que han sido recomendados: TP/TP+FN
  # FPR: False positive rate
  # % de artículos no comprados que han sido recomendados: FP/FP+TN
plot(results, annotate=T, main="Curva ROC")
  # Precision: % de artículos recomendados que han sido comprados: FP/TP+FP
  # Recall (TPR): % de artículos comprados que han sido recomendados: TP/TP+FN
  # Si el % de artículos comprados es pequeño, la Precision serí baja. Oscila 0-1.
plot(results, "prec/rec", annotate=T, main="Precision/Recall")

### Identificar el mejor modelo
models_to_evaluate <- list(
  "aleatorios" = list(name="RANDOM", param=NULL),
  "user-based-cos" = list(name="UBCF", param=list(normalize = "Z-score",method="Cosine")),
  "used-based-cor" = list(name="UBCF", param=list(normalize = "center",method="pearson")),
  "item-based-cos" = list(name="IBCF", param=list(normalize = "Z-score",method="Cosine")),
  "item-based-cor" = list(name="IBCF", param=list(normalize = "center",method="pearson")))

list_results <- evaluate(eval_sets,
                         models_to_evaluate,
                         type = "topNList",
                         n=c(1, 5, seq(10,100,10)))
class(list_results)

avg_matrices <- lapply(list_results,avg)
avg_matrices$aleatorios
plot(list_results, annotate = 1, legend = "topleft")
title(("curva ROC"))
plot(list_results, "prec/rec", annotate = 1, legend = "bottomright")
title("Precision/Recall")

### Optimizar un parámetro numérico
# ¿Cómo optimizamos k?

vector_k <- c(5,10,20,30,40)

models_to_evaluate <- lapply(vector_k, function(k){
  list(name="IBCF", param=list(method="Cosine", k = k))
})

names(models_to_evaluate) <- paste0("IBCF_k_", vector_k)

list_results <- evaluate(eval_sets,
                         models_to_evaluate,
                         n=c(1, 5, seq(10,100,10)))

plot(list_results, annotate = 1, legend = "topleft")
plot(list_results, annotate = 1, legend = "bottomright") # el k ´óptimo es k = 40
title("curva ROC")

plot(list_results, "prec/rec", annotate = 1, legend = "topleft")
plot(list_results, "prec/rec", annotate = 1, legend = "bottomright") # el k ´óptimo es k = 30/40
title("Precision/Recall")

