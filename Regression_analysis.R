# =============================================================================
#  ESO301 — Applied Statistics for Environmental Engineering
#  REGRESSION ANALYSIS
#  Target: pr (Precipitation) | Predictors: mrso, mrros, evspsbl
#  Models: MLR | Polynomial | Random Forest | XGBoost
#  Mahanadi Basin | EC-Earth3-Veg | SSP1-2.6 | Group 3
# =============================================================================

# ── 0. Packages ───────────────────────────────────────────────────────────────
required_reg <- c("ggplot2", "dplyr", "tidyr", "gridExtra", "grid",
                  "scales", "randomForest", "xgboost", "car", "reshape2")
for (pkg in required_reg) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}
library(ggplot2); library(dplyr); library(tidyr)
library(gridExtra); library(grid); library(scales)
library(randomForest); library(xgboost)
library(car); library(reshape2)

# ── 1. Paths ──────────────────────────────────────────────────────────────────
input_csv  <- "~/Desktop/mahanadi_climate_data.csv"
output_dir <- "~/Desktop"

# ── 2. Load & Prepare Data ────────────────────────────────────────────────────
df_raw      <- read.csv(input_csv, stringsAsFactors = FALSE)
df_raw$time <- as.Date(paste0(df_raw$time, "-01"))
df_raw      <- df_raw[order(df_raw$time), ]

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
train <- df_raw[df_raw$time <= as.Date("2031-12-01"), ]
test  <- df_raw[df_raw$time >  as.Date("2031-12-01"), ]

cat(sprintf("Train: %d months (%s to %s)\n", nrow(train), min(train$time), max(train$time)))
cat(sprintf("Test : %d months (%s to %s)\n", nrow(test),  min(test$time),  max(test$time)))

X_train <- train[, c("mrso", "mrros", "evspsbl")]
y_train <- train$pr
X_test  <- test[,  c("mrso", "mrros", "evspsbl")]
y_test  <- test$pr

# ── 4. Metrics Helper ─────────────────────────────────────────────────────────
calc_metrics <- function(actual, predicted, label) {
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  data.frame(
    Model = label,
    R2    = round(1 - ss_res / ss_tot, 4),
    RMSE  = round(sqrt(mean((actual - predicted)^2)), 4),
    MAE   = round(mean(abs(actual - predicted)), 4)
  )
}

# ── 5. Dark Theme ─────────────────────────────────────────────────────────────
bg_dark  <- "#0D1117"; bg_panel <- "#161B22"; bg_strip <- "#1C2333"
col_text <- "#E6EDF3"; col_sub  <- "#8B949E"; col_grid <- "#21262D"
col_bord <- "#30363D"
caption_text <- paste0("Mahanadi Basin  \u00b7  EC-Earth3-Veg  \u00b7  ",
                       "SSP1-2.6  \u00b7  Group 3  |  ESO301 Applied Statistics")

dark_theme <- theme(
  plot.background   = element_rect(fill = bg_dark,  color = NA),
  panel.background  = element_rect(fill = bg_panel, color = NA),
  panel.grid.major  = element_line(color = col_grid, linewidth = 0.4),
  panel.grid.minor  = element_blank(),
  panel.border      = element_rect(color = col_bord, fill = NA, linewidth = 0.6),
  axis.text         = element_text(color = col_sub,  size = 9),
  axis.title        = element_text(color = col_text, size = 10, face = "bold"),
  axis.ticks        = element_line(color = col_bord),
  plot.title        = element_text(color = col_text, size = 14, face = "bold",
                                   hjust = 0, margin = margin(b = 4)),
  plot.subtitle     = element_text(color = col_sub,  size = 9,  hjust = 0,
                                   margin = margin(b = 12), lineheight = 1.4),
  plot.caption      = element_text(color = col_sub,  size = 7.5,
                                   hjust = 0, margin = margin(t = 8)),
  plot.margin       = margin(20, 20, 14, 20),
  legend.background = element_rect(fill = bg_panel, color = col_bord),
  legend.key        = element_rect(fill = bg_panel, color = NA),
  legend.text       = element_text(color = col_text, size = 9),
  legend.title      = element_text(color = col_sub,  size = 9, face = "bold"),
  strip.background  = element_rect(fill = bg_strip,  color = col_bord),
  strip.text        = element_text(color = col_text, size = 9, face = "bold")
)

save_plot <- function(filename, width = 13, height = 7.5) {
  ggsave(file.path(output_dir, filename),
         width = width, height = height, dpi = 180, bg = bg_dark)
  message("Saved: ", filename)
}

model_colors <- c(
  "Actual"       = "#FFFFFF",
  "MLR"          = "#00B4D8",
  "Polynomial"   = "#FFB703",
  "RandomForest" = "#06D6A0",
  "XGBoost"      = "#FF6B6B"
)

# =============================================================================
#  MODEL 1 — MULTIPLE LINEAR REGRESSION
# =============================================================================
mlr_model      <- lm(pr ~ mrso + mrros + evspsbl, data = train)
mlr_train_pred <- predict(mlr_model, newdata = train)
mlr_test_pred  <- predict(mlr_model, newdata = test)
mlr_metrics    <- calc_metrics(y_test, mlr_test_pred, "MLR")

cat("\n── MLR Summary ──\n")
print(summary(mlr_model))
cat("\n── VIF ──\n")
print(vif(mlr_model))

# =============================================================================
#  MODEL 2 — POLYNOMIAL REGRESSION
# =============================================================================
poly2_model     <- lm(pr ~ poly(mrso,2) + poly(mrros,2) + poly(evspsbl,2), data = train)
poly3_model     <- lm(pr ~ poly(mrso,3) + poly(mrros,3) + poly(evspsbl,3), data = train)
poly2_test_pred <- predict(poly2_model, newdata = test)
poly3_test_pred <- predict(poly3_model, newdata = test)
m2 <- calc_metrics(y_test, poly2_test_pred, "Poly-2")
m3 <- calc_metrics(y_test, poly3_test_pred, "Poly-3")

cat("\n── Polynomial Degree Selection ──\n")
print(rbind(m2, m3))

if (m2$R2 >= m3$R2) {
  poly_model      <- poly2_model
  poly_test_pred  <- poly2_test_pred
  poly_train_pred <- predict(poly2_model, newdata = train)
  poly_degree     <- 2
} else {
  poly_model      <- poly3_model
  poly_test_pred  <- poly3_test_pred
  poly_train_pred <- predict(poly3_model, newdata = train)
  poly_degree     <- 3
}
poly_label   <- paste0("Polynomial (deg ", poly_degree, ")")
poly_metrics <- calc_metrics(y_test, poly_test_pred, poly_label)
cat(sprintf("\nSelected degree: %d  |  Test R² = %.4f\n", poly_degree, poly_metrics$R2))

# Raw polynomial coefficients (interpretable equation)
poly_raw_model <- lm(
  pr ~ I(mrso) + I(mrso^2) + I(mrso^3) +
    I(mrros) + I(mrros^2) + I(mrros^3) +
    I(evspsbl) + I(evspsbl^2) + I(evspsbl^3),
  data = train
)
cat("\n── Polynomial Raw Coefficients ──\n")
coef_raw <- coef(poly_raw_model)
cat(sprintf("pr = %.6f\n",           coef_raw["(Intercept)"]))
cat(sprintf("   + (%.6f) x mrso\n",      coef_raw["I(mrso)"]))
cat(sprintf("   + (%.8f) x mrso2\n",     coef_raw["I(mrso^2)"]))
cat(sprintf("   + (%.10f) x mrso3\n",    coef_raw["I(mrso^3)"]))
cat(sprintf("   + (%.6f) x mrros\n",     coef_raw["I(mrros)"]))
cat(sprintf("   + (%.6f) x mrros2\n",    coef_raw["I(mrros^2)"]))
cat(sprintf("   + (%.8f) x mrros3\n",    coef_raw["I(mrros^3)"]))
cat(sprintf("   + (%.6f) x evspsbl\n",   coef_raw["I(evspsbl)"]))
cat(sprintf("   + (%.6f) x evspsbl2\n",  coef_raw["I(evspsbl^2)"]))
cat(sprintf("   + (%.8f) x evspsbl3\n",  coef_raw["I(evspsbl^3)"]))

# =============================================================================
#  MODEL 3 — RANDOM FOREST
# =============================================================================
set.seed(42)
rf_model <- randomForest(
  x = X_train, y = y_train,
  ntree = 500, mtry = 2, importance = TRUE
)
rf_train_pred <- predict(rf_model, newdata = X_train)
rf_test_pred  <- predict(rf_model, newdata = X_test)
rf_metrics    <- calc_metrics(y_test, rf_test_pred, "Random Forest")

cat("\n── Random Forest ──\n")
print(rf_model)
cat("\n── RF Variable Importance ──\n")
print(importance(rf_model))

# =============================================================================
#  MODEL 4 — XGBOOST
# =============================================================================
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest  <- xgb.DMatrix(data = as.matrix(X_test),  label = y_test)

set.seed(42)
xgb_model <- xgb.train(
  params = list(
    objective        = "reg:squarederror",
    max_depth        = 4,
    eta              = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3
  ),
  data                  = dtrain,
  nrounds               = 300,
  evals                 = list(train = dtrain, test = dtest),
  verbose               = 0,
  early_stopping_rounds = 30
)
xgb_train_pred <- predict(xgb_model, dtrain)
xgb_test_pred  <- predict(xgb_model, dtest)
xgb_metrics    <- calc_metrics(y_test, xgb_test_pred, "XGBoost")
xgb_importance <- xgb.importance(feature_names = colnames(X_train), model = xgb_model)

cat("\n── XGBoost Best Iteration:", xgb_model$best_iteration, "──\n")
cat("\n── XGBoost Feature Importance ──\n")
print(xgb_importance)

# =============================================================================
#  FINAL METRICS SUMMARY
# =============================================================================
all_metrics <- rbind(mlr_metrics, poly_metrics, rf_metrics, xgb_metrics)
all_metrics$Model <- factor(all_metrics$Model, levels = all_metrics$Model)

cat("\n", strrep("=", 55), "\n")
cat("  TEST SET PERFORMANCE (2032-2035)\n")
cat(strrep("=", 55), "\n")
print(all_metrics)

# =============================================================================
#  PLOT R1 — Time Series Prediction
# =============================================================================
pred_df <- data.frame(
  time = test$time, Actual = y_test,
  MLR = mlr_test_pred, Polynomial = poly_test_pred,
  RandomForest = rf_test_pred, XGBoost = xgb_test_pred
)
pred_long       <- pivot_longer(pred_df, -time, names_to = "Model", values_to = "pr")
pred_long$Model <- factor(pred_long$Model,
                          levels = c("Actual","MLR","Polynomial","RandomForest","XGBoost"))

monsoon_df <- do.call(rbind, lapply(2032:2035, function(yr) {
  data.frame(xmin = as.Date(paste0(yr,"-06-01")),
             xmax = as.Date(paste0(yr,"-09-30")),
             ymin = -Inf, ymax = Inf)
}))

r2_labels <- c(
  "Actual"       = "Actual pr",
  "MLR"          = sprintf("MLR  (R\u00b2=%.3f)", mlr_metrics$R2),
  "Polynomial"   = sprintf("%s  (R\u00b2=%.3f)", poly_label, poly_metrics$R2),
  "RandomForest" = sprintf("Random Forest  (R\u00b2=%.3f)", rf_metrics$R2),
  "XGBoost"      = sprintf("XGBoost  (R\u00b2=%.3f)", xgb_metrics$R2)
)
line_types  <- c("Actual"="solid","MLR"="longdash","Polynomial"="dotted",
                 "RandomForest"="longdash","XGBoost"="longdash")
line_widths <- c("Actual"=1.8,"MLR"=1.0,"Polynomial"=1.0,
                 "RandomForest"=1.0,"XGBoost"=1.0)

ggplot() +
  geom_rect(data = monsoon_df,
            aes(xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax),
            fill="#00B4D8", alpha=0.07, inherit.aes=FALSE) +
  annotate("text", x=as.Date(paste0(2032:2035,"-07-15")), y=Inf,
           label="Monsoon", color="#00B4D8", size=3, vjust=1.5, alpha=0.8) +
  geom_line(data=pred_long,
            aes(x=time,y=pr,color=Model,linetype=Model,linewidth=Model)) +
  scale_color_manual(values=model_colors, labels=r2_labels, name=NULL) +
  scale_linetype_manual(values=line_types, labels=r2_labels, name=NULL) +
  scale_linewidth_manual(values=line_widths, labels=r2_labels, name=NULL) +
  scale_x_date(date_breaks="3 months", date_labels="%b '%y") +
  labs(title    = "Precipitation Prediction \u2014 Test Period 2032\u20132035",
       subtitle = paste0("Mahanadi Basin  \u00b7  EC-Earth3-Veg  \u00b7  SSP1-2.6  \u00b7  Group 3\n",
                         "Shaded regions indicate monsoon season (Jun\u2013Sep)"),
       x=NULL, y="Precipitation (mm / month)", caption=caption_text) +
  dark_theme +
  theme(axis.text.x=element_text(angle=30,hjust=1),
        legend.position="bottom", legend.key.width=unit(2,"cm"))
save_plot("plot_R1_timeseries_prediction.png", width=14, height=7.5)

# =============================================================================
#  PLOT R2 — Actual vs Predicted Scatter
# =============================================================================
scatter_df <- data.frame(
  Actual    = rep(y_test, 4),
  Predicted = c(mlr_test_pred, poly_test_pred, rf_test_pred, xgb_test_pred),
  Model     = factor(rep(c("MLR",poly_label,"Random Forest","XGBoost"),
                         each=length(y_test)),
                     levels=c("MLR",poly_label,"Random Forest","XGBoost"))
)
scatter_colors   <- setNames(c("#00B4D8","#FFB703","#06D6A0","#FF6B6B"),
                             c("MLR",poly_label,"Random Forest","XGBoost"))
r2_vec           <- c(mlr_metrics$R2,poly_metrics$R2,rf_metrics$R2,xgb_metrics$R2)
scatter_labeller <- setNames(sprintf("%s\nR\u00b2 = %.3f",
                                     levels(scatter_df$Model), r2_vec),
                             levels(scatter_df$Model))

ggplot(scatter_df, aes(x=Actual, y=Predicted)) +
  geom_abline(slope=1, intercept=0, color=col_sub, linewidth=0.7, linetype="dashed") +
  geom_point(aes(color=Model), alpha=0.65, size=2.2) +
  geom_smooth(aes(color=Model), method="lm", se=TRUE,
              linewidth=1, fill=bg_strip, alpha=0.25) +
  facet_wrap(~Model, ncol=2, labeller=as_labeller(scatter_labeller)) +
  scale_color_manual(values=scatter_colors) +
  labs(title    = "Actual vs Predicted \u2014 Scatter Plot by Model",
       subtitle = "Dashed line = perfect prediction (slope = 1)  \u00b7  Closer points = better fit",
       x="Actual Precipitation (mm / month)",
       y="Predicted Precipitation (mm / month)", caption=caption_text) +
  dark_theme + theme(legend.position="none")
save_plot("plot_R2_actual_vs_predicted.png", width=12, height=9)

# =============================================================================
#  PLOT R3 — Model Performance Metrics
# =============================================================================
metrics_long        <- pivot_longer(all_metrics,-Model,names_to="Metric",values_to="Value")
metrics_long$Metric <- factor(metrics_long$Metric, levels=c("R2","RMSE","MAE"))
metric_labeller     <- c(R2="R\u00b2  (higher is better)",
                         RMSE="RMSE  (lower is better)",
                         MAE="MAE  (lower is better)")
bar_colors          <- setNames(c("#00B4D8","#FFB703","#06D6A0","#FF6B6B"),
                                levels(all_metrics$Model))

ggplot(metrics_long, aes(x=Model, y=Value, fill=Model)) +
  geom_col(width=0.6, alpha=0.9) +
  geom_text(aes(label=sprintf("%.3f",Value), vjust=ifelse(Value>=0,-0.5,1.4)),
            color=col_text, size=3.2, fontface="bold") +
  facet_wrap(~Metric, scales="free_y", labeller=as_labeller(metric_labeller)) +
  scale_fill_manual(values=bar_colors) +
  scale_y_continuous(expand=expansion(mult=c(0.05,0.18))) +
  labs(title    = "Model Performance Comparison \u2014 Test Set Metrics",
       subtitle = "All metrics computed on unseen test data (2032\u20132035)",
       x=NULL, y="Metric Value", caption=caption_text) +
  dark_theme +
  theme(axis.text.x=element_text(angle=20,hjust=1,size=8.5),
        legend.position="none")
save_plot("plot_R3_model_metrics.png", width=13, height=6)

# =============================================================================
#  PLOT R4 — MLR Residual Diagnostics
# =============================================================================
resid_df <- data.frame(
  time=train$time, fitted=mlr_train_pred,
  residual=resid(mlr_model), std_resid=rstandard(mlr_model)
)
resid_df$sqrt_abs <- sqrt(abs(resid_df$std_resid))

p_rv <- ggplot(resid_df, aes(x=fitted, y=residual)) +
  geom_hline(yintercept=0, color=col_sub, linetype="dashed") +
  geom_point(color="#00B4D8", alpha=0.7, size=2) +
  geom_smooth(method="loess", se=FALSE, color="#FF6B6B", linewidth=1, formula=y~x) +
  labs(title="Residuals vs Fitted", x="Fitted Values", y="Residuals") +
  dark_theme + theme(plot.margin=margin(10,10,10,10))

qq_data <- qqnorm(resid_df$std_resid, plot.it=FALSE)
qq_df   <- data.frame(theoretical=qq_data$x, sample=qq_data$y)
p_qq <- ggplot(qq_df, aes(x=theoretical, y=sample)) +
  geom_abline(slope=1, intercept=0, color=col_sub, linetype="dashed") +
  geom_point(color="#FFB703", alpha=0.8, size=2) +
  labs(title="Normal Q-Q Plot", x="Theoretical Quantiles", y="Standardised Residuals") +
  dark_theme + theme(plot.margin=margin(10,10,10,10))

p_sl <- ggplot(resid_df, aes(x=fitted, y=sqrt_abs)) +
  geom_point(color="#06D6A0", alpha=0.7, size=2) +
  geom_smooth(method="loess", se=FALSE, color="#FF6B6B", linewidth=1, formula=y~x) +
  labs(title="Scale-Location  (Homoscedasticity)",
       x="Fitted Values", y="\u221a|Standardised Residuals|") +
  dark_theme + theme(plot.margin=margin(10,10,10,10))

p_rt <- ggplot(resid_df, aes(x=time, y=residual)) +
  geom_hline(yintercept=0, color=col_sub, linetype="dashed") +
  geom_line(color=col_sub, linewidth=0.5, alpha=0.5) +
  geom_point(color="#FF6B6B", alpha=0.7, size=1.8) +
  scale_x_date(date_breaks="1 year", date_labels="%Y") +
  labs(title="Residuals over Time  (Autocorrelation Check)", x=NULL, y="Residuals") +
  dark_theme + theme(plot.margin=margin(10,10,10,10))

resid_grid <- arrangeGrob(p_rv, p_qq, p_sl, p_rt, ncol=2,
                          top    = textGrob("MLR Residual Diagnostics \u2014 Assumption Checks",
                                            gp=gpar(col=col_text, fontsize=14, fontface="bold")),
                          bottom = textGrob(caption_text, gp=gpar(col=col_sub, fontsize=7.5)))
ggsave(file.path(output_dir,"plot_R4_mlr_residuals.png"),
       plot=resid_grid, width=13, height=9, dpi=180, bg=bg_dark)
message("Saved: plot_R4_mlr_residuals.png")

# =============================================================================
#  PLOT R5 — Variable Importance: RF + XGBoost
# =============================================================================
rf_imp_df  <- data.frame(Variable=rownames(importance(rf_model)),
                         Importance=as.numeric(importance(rf_model)[,"%IncMSE"]),
                         Model="Random Forest")
xgb_imp_df <- data.frame(Variable=xgb_importance$Feature,
                         Importance=xgb_importance$Gain*100,
                         Model="XGBoost")
imp_df       <- rbind(rf_imp_df, xgb_imp_df)
imp_df$Model <- factor(imp_df$Model, levels=c("Random Forest","XGBoost"))
var_colors   <- c(mrso="#FFB703", mrros="#06D6A0", evspsbl="#FF6B6B")

ggplot(imp_df, aes(x=reorder(Variable,Importance), y=Importance, fill=Variable)) +
  geom_col(width=0.55, alpha=0.9) +
  geom_text(aes(label=sprintf("%.1f",Importance), hjust=-0.15),
            color=col_text, size=3.5, fontface="bold") +
  coord_flip() +
  facet_wrap(~Model, scales="free_x") +
  scale_fill_manual(values=var_colors) +
  scale_y_continuous(expand=expansion(mult=c(0,0.20))) +
  labs(title    = "Variable Importance \u2014 Random Forest vs XGBoost",
       subtitle = paste0("RF: %IncMSE = % increase in error when variable is randomly permuted\n",
                         "XGBoost: Gain = % improvement in loss brought by each feature across all splits"),
       x="Predictor Variable", y="Importance Score", caption=caption_text) +
  dark_theme + theme(legend.position="none")
save_plot("plot_R5_variable_importance.png", width=12, height=6)

# =============================================================================
#  PLOT R6 — VIF Multicollinearity Check
# =============================================================================
vif_vals <- vif(mlr_model)
vif_df   <- data.frame(Variable=names(vif_vals), VIF=as.numeric(vif_vals),
                       stringsAsFactors=FALSE)
vif_df$Status <- ifelse(vif_df$VIF<5, "Low (<5)",
                        ifelse(vif_df$VIF<10, "Moderate (5\u201310)",
                               "High (>10 \u2014 problematic)"))
vif_colors <- c("Low (<5)"="#06D6A0",
                "Moderate (5\u201310)"="#FFB703",
                "High (>10 \u2014 problematic)"="#FF6B6B")

ggplot(vif_df, aes(x=reorder(Variable,VIF), y=VIF, fill=Status)) +
  geom_col(width=0.5, alpha=0.9) +
  geom_hline(yintercept=10, color="#FF6B6B", linetype="dashed", linewidth=0.8) +
  geom_hline(yintercept=5,  color="#FFB703", linetype="dashed", linewidth=0.8) +
  annotate("text",x=0.6,y=10.4,label="Problem threshold (VIF = 10)",
           color="#FF6B6B",size=3,hjust=0) +
  annotate("text",x=0.6,y=5.4,label="Warning threshold (VIF = 5)",
           color="#FFB703",size=3,hjust=0) +
  geom_text(aes(label=sprintf("%.2f",VIF),vjust=-0.5),
            color=col_text,size=4,fontface="bold") +
  coord_flip() +
  scale_fill_manual(values=vif_colors, name="Multicollinearity") +
  scale_y_continuous(limits=c(0,max(vif_df$VIF)*1.35)) +
  labs(title    = "VIF \u2014 Multicollinearity Check for MLR",
       subtitle = "VIF < 5 = acceptable  \u00b7  VIF 5\u201310 = moderate  \u00b7  VIF > 10 = problematic",
       x="Predictor", y="Variance Inflation Factor (VIF)", caption=caption_text) +
  dark_theme + theme(legend.position="right")
save_plot("plot_R6_vif_check.png", width=11, height=5.5)

# =============================================================================
#  PLOT R7 — Train vs Test R² (Overfitting Check)
# =============================================================================
train_metrics <- rbind(
  calc_metrics(y_train, mlr_train_pred,  "MLR"),
  calc_metrics(y_train, poly_train_pred, poly_label),
  calc_metrics(y_train, rf_train_pred,   "Random Forest"),
  calc_metrics(y_train, xgb_train_pred,  "XGBoost")
)
overfit_df <- data.frame(
  Model = factor(rep(levels(all_metrics$Model),2), levels=levels(all_metrics$Model)),
  Split = rep(c("Train (2026\u20132031)","Test (2032\u20132035)"), each=nrow(all_metrics)),
  R2    = c(train_metrics$R2, all_metrics$R2)
)
split_colors2 <- c("Train (2026\u20132031)"="#00B4D8","Test (2032\u20132035)"="#FF6B6B")

ggplot(overfit_df, aes(x=Model, y=R2, fill=Split)) +
  geom_col(position=position_dodge(width=0.7), width=0.65, alpha=0.9) +
  geom_text(aes(label=sprintf("%.3f",R2)),
            position=position_dodge(width=0.7),
            vjust=-0.5, color=col_text, size=3.3, fontface="bold") +
  scale_fill_manual(values=split_colors2, name="Dataset") +
  scale_y_continuous(limits=c(0,1.12), breaks=seq(0,1,0.2),
                     labels=number_format(accuracy=0.01)) +
  labs(title    = "Train vs Test R\u00b2 \u2014 Overfitting Check",
       subtitle = "Large gap = overfitting  \u00b7  Ideal: both values close together",
       x=NULL, y="R\u00b2", caption=caption_text) +
  dark_theme +
  theme(axis.text.x=element_text(angle=15,hjust=1,size=9),
        legend.position="top")
save_plot("plot_R7_overfitting_check.png", width=11, height=6.5)

# =============================================================================
#  SPLIT SENSITIVITY — 60:40 vs 90:10
# =============================================================================
train_90 <- df_raw[df_raw$time <= as.Date("2034-12-01"), ]
test_10  <- df_raw[df_raw$time >  as.Date("2034-12-01"), ]
X_train_90 <- train_90[, c("mrso","mrros","evspsbl")]
y_train_90 <- train_90$pr
X_test_10  <- test_10[,  c("mrso","mrros","evspsbl")]
y_test_10  <- test_10$pr

mlr_90      <- lm(pr ~ mrso + mrros + evspsbl, data=train_90)
mlr_90_m    <- calc_metrics(y_test_10, predict(mlr_90,newdata=test_10), "MLR")

poly_90     <- lm(pr ~ poly(mrso,poly_degree)+poly(mrros,poly_degree)+
                    poly(evspsbl,poly_degree), data=train_90)
poly_90_m   <- calc_metrics(y_test_10, predict(poly_90,newdata=test_10),
                            paste0("Polynomial (deg ",poly_degree,")"))

set.seed(42)
rf_90       <- randomForest(x=X_train_90, y=y_train_90, ntree=500, mtry=2)
rf_90_m     <- calc_metrics(y_test_10, predict(rf_90,newdata=X_test_10), "Random Forest")

dtrain_90   <- xgb.DMatrix(data=as.matrix(X_train_90), label=y_train_90)
dtest_10    <- xgb.DMatrix(data=as.matrix(X_test_10),  label=y_test_10)
set.seed(42)
xgb_90      <- xgb.train(
  params=list(objective="reg:squarederror",max_depth=4,eta=0.05,
              subsample=0.8,colsample_bytree=0.8),
  data=dtrain_90, nrounds=300,
  evals=list(train=dtrain_90,test=dtest_10),
  verbose=0, early_stopping_rounds=30)
xgb_90_m    <- calc_metrics(y_test_10, predict(xgb_90,dtest_10), "XGBoost")

metrics_90_10 <- rbind(mlr_90_m, poly_90_m, rf_90_m, xgb_90_m)
comparison_df <- data.frame(
  Model      = as.character(all_metrics$Model),
  R2_60_40   = all_metrics$R2,
  R2_90_10   = metrics_90_10$R2,
  RMSE_60_40 = all_metrics$RMSE,
  RMSE_90_10 = metrics_90_10$RMSE
)
cat("\n── Split Comparison ──\n")
print(comparison_df)

# PLOT R9 — Split R² Comparison
comp_long <- data.frame(
  Model = rep(comparison_df$Model, 2),
  Split = rep(c("60:40  (Train 2026\u20132031)","90:10  (Train 2026\u20132034)"),
              each=nrow(comparison_df)),
  R2    = c(comparison_df$R2_60_40, comparison_df$R2_90_10)
)
comp_long$Model <- factor(comp_long$Model, levels=comparison_df$Model)
comp_long$Split <- factor(comp_long$Split,
                          levels=c("60:40  (Train 2026\u20132031)",
                                   "90:10  (Train 2026\u20132034)"))
split_col <- c("60:40  (Train 2026\u20132031)"="#00B4D8",
               "90:10  (Train 2026\u20132034)"="#FF6B6B")

ggplot(comp_long, aes(x=Model, y=R2, fill=Split)) +
  geom_col(position=position_dodge(width=0.7), width=0.65, alpha=0.9) +
  geom_text(aes(label=sprintf("%.3f",R2)),
            position=position_dodge(width=0.7),
            vjust=-0.5, color=col_text, size=3.3, fontface="bold") +
  geom_hline(yintercept=0.95, color=col_sub, linetype="dashed",
             linewidth=0.5, alpha=0.5) +
  scale_fill_manual(values=split_col, name="Train:Test Split") +
  scale_y_continuous(limits=c(0,1.12), breaks=seq(0,1,0.1),
                     labels=number_format(accuracy=0.01)) +
  labs(title    = "Split Sensitivity \u2014 60:40 vs 90:10 Comparison",
       subtitle = paste0("60:40 = Train 2026\u20132031 (72 months) | Test 2032\u20132035 (48 months)\n",
                         "90:10 = Train 2026\u20132034 (108 months) | Test 2035 only (12 months)"),
       x=NULL, y="Test R\u00b2", caption=caption_text) +
  dark_theme +
  theme(axis.text.x=element_text(angle=15,hjust=1,size=9),
        legend.position="top", plot.subtitle=element_text(lineheight=1.5))
ggsave(file.path(output_dir,"plot_R9_split_comparison.png"),
       width=13, height=7, dpi=180, bg=bg_dark)
message("Saved: plot_R9_split_comparison.png")

# PLOT R10 — Split RMSE Comparison
rmse_long <- data.frame(
  Model = rep(comparison_df$Model, 2),
  Split = rep(c("60:40  (Train 2026\u20132031)","90:10  (Train 2026\u20132034)"),
              each=nrow(comparison_df)),
  RMSE  = c(comparison_df$RMSE_60_40, comparison_df$RMSE_90_10)
)
rmse_long$Model <- factor(rmse_long$Model, levels=comparison_df$Model)
rmse_long$Split <- factor(rmse_long$Split,
                          levels=c("60:40  (Train 2026\u20132031)",
                                   "90:10  (Train 2026\u20132034)"))

ggplot(rmse_long, aes(x=Model, y=RMSE, fill=Split)) +
  geom_col(position=position_dodge(width=0.7), width=0.65, alpha=0.9) +
  geom_text(aes(label=sprintf("%.2f",RMSE)),
            position=position_dodge(width=0.7),
            vjust=-0.5, color=col_text, size=3.3, fontface="bold") +
  scale_fill_manual(values=split_col, name="Train:Test Split") +
  scale_y_continuous(expand=expansion(mult=c(0,0.15))) +
  labs(title    = "Split Sensitivity \u2014 RMSE Comparison (60:40 vs 90:10)",
       subtitle = "Lower RMSE = better  \u00b7  Consistent RMSE across splits confirms stability",
       x=NULL, y="RMSE (mm / month)", caption=caption_text) +
  dark_theme +
  theme(axis.text.x=element_text(angle=15,hjust=1,size=9),
        legend.position="top")
ggsave(file.path(output_dir,"plot_R10_split_rmse.png"),
       width=13, height=7, dpi=180, bg=bg_dark)
message("Saved: plot_R10_split_rmse.png")

# =============================================================================
#  FINAL SUMMARY
# =============================================================================
cat("\n", strrep("=", 62), "\n")
cat("  REGRESSION ANALYSIS COMPLETE\n")
cat(strrep("=", 62), "\n")
for (f in c("plot_R1_timeseries_prediction.png",
            "plot_R2_actual_vs_predicted.png",
            "plot_R3_model_metrics.png",
            "plot_R4_mlr_residuals.png",
            "plot_R5_variable_importance.png",
            "plot_R6_vif_check.png",
            "plot_R7_overfitting_check.png",
            "plot_R9_split_comparison.png",
            "plot_R10_split_rmse.png")) {
  cat("  \u2713", f, "\n")
}
cat(strrep("=", 62), "\n")
cat("\n  FINAL MODEL PERFORMANCE (Test Set)\n\n")
print(all_metrics)