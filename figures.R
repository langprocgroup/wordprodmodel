rm(list=ls())
setwd("/Users/canjo/projects/control")
library(tidyverse)
library(stringr)
library(forcats)
library(pracma)
library(latex2exp)

logistic = function(x, k, x0) {
  1 / (1 + exp(-k*(x-x0)))
}

interference = function(gammas, qs, N=10, steps=100) {
  x = linspace(0, N, steps)
  expand.grid(gammas, qs, x) %>%
    rename(gamma=Var1, q=Var2, d=Var3) %>%
    group_by(gamma, q) %>%
      mutate(p=logistic(d, gamma, 1/gamma * log((1-q)/q))) %>%
      ungroup()
}

G = c(.5, 1, 2)
Q = c(.5, .7, .9)

interference(G, Q, steps=1000) %>% 
  ggplot(aes(x=d, y=p, linetype=as.factor(gamma), color=as.factor(q))) +
    theme_minimal() +
    theme(legend.position = "none") +
    geom_line(aes(group=interaction(gamma, q))) +
    scale_linetype_manual(values=c("dotted", "dashed", "solid")) +
    xlab(TeX("Distance to distractor $d(a_s, a_d)$")) +
    ylab(TeX("$q(a_s | s)$")) + 
    ylim(NA,1) +
    ggtitle(TeX("Probability of action $a_s$ in state $s$"))

ggsave("prob_logistic.pdf", width=4, height=4)

interference(G, Q, steps=1000) %>% 
  mutate(gamma=format(gamma, nsmall=1)) %>%
  ggplot(aes(x=d, y=-log(p,2), linetype=as.factor(gamma), color=as.factor(q))) +
    theme_minimal() +
    theme(legend.position = "none") +
    geom_line(aes(group=interaction(gamma, q))) +
    scale_linetype_manual(values=c("dotted", "dashed", "solid")) +
    xlab(TeX("Distance to distractor $d(a_s, a_d)$")) +
    ylab(TeX("-log $q(a_s|s)$ (bits)")) + 
    labs(linetype=TeX("$\\gamma$"), color=TeX("Prior $q(a_s)$")) +
    ggtitle(TeX("Decision cost"))

ggsave("control_logistic.pdf", width=4, height=4)

interference(G,Q, steps=1000) %>% 
  mutate(gamma=format(gamma, nsmall=1)) %>%
  ggplot(aes(x=d, y=log(p,2)-log(q,2), linetype=as.factor(gamma), color=as.factor(q))) +
    theme_minimal() +
    geom_line(aes(group=interaction(gamma, q))) +
    labs(linetype=TeX("$\\gamma$"), color=TeX("Prior $q(a_s)$")) +
    scale_linetype_manual(values=c("dotted", "dashed", "solid")) +
    xlab(TeX("Distance to distractor $d(a_s, a_d)$")) +
    ylab(TeX("log q(a_s|s)/q(a_s) (bits)")) + 
    ggtitle(TeX("Computation cost"))

ggsave("computation_logistic.pdf", width=5, height=4)

basic_sim = read_csv("basic_simulation.csv") %>% select(-X1)
neutral_sim = read_csv("neutral_simulation.csv") %>% filter(d == 2)

basic_sim %>% 
  gather(measure, value, -goal, -d) %>% 
  filter(measure %in% c("h", "s")) %>%
  mutate(measure=if_else(measure == "h", "Decision cost", "Computation cost")) %>%
  mutate(value=value/log(2)) %>%
  ggplot(aes(x=d, y=value, color=goal)) + 
    geom_line() + 
    facet_wrap(~measure, scale="free_y") + 
    theme_minimal() +
    theme(legend.position="bottom") +
    ylab("Bits") +
    xlab("Distractor distance") 

ggsave("basic_simulation.pdf", width=6, height=3)

reverse_sim = read_csv("reverse_simulation.csv") %>% select(-X1)
reverse_sim %>% 
  gather(measure, value, -goal, -d, -g) %>% 
  filter(measure %in% c("h", "s")) %>%
  mutate(measure=if_else(measure == "h", "Decision cost", "Computation cost")) %>%
  mutate(value=value/log(2)) %>%
  mutate(`p(g=name)`=as.factor(g)) %>%
  filter(measure == "Decision cost") %>%
  ggplot(aes(x=d, y=value, color=goal, linetype=`p(g=name)`)) + 
    geom_line() + 
    #facet_wrap(~measure, scale="free_y") + 
    theme_minimal() +
    #theme(legend.position="bottom") +
    ylab("Decision cost (bits)") +
    xlab("Distractor distance") +
    xlim(NA, 5)

ggsave("reverse_simulation.pdf", width=5, height=4)

rp = read_csv("pwi_roelofs_piai2017.csv") %>%
  gather(measure, value, -condition, -variable) %>% 
  spread(variable, value) 

filter(rp, measure == "M") %>% 
  ggplot(aes(x=condition, y=mean, ymin=mean-1.96*se, ymax=mean+1.96*se, color=condition)) + 
    geom_point() + 
    geom_errorbar()

filter(rp, measure == "mu") %>% 
  ggplot(aes(x=condition, y=mean, ymin=mean-1.96*se, ymax=mean+1.96*se, color=condition)) + 
    geom_point() + 
    geom_errorbar()

filter(rp, measure == "tau") %>% 
  ggplot(aes(x=condition, y=mean, ymin=mean-1.96*se, ymax=mean+1.96*se, color=condition)) + 
  geom_point() + 
  geom_errorbar()

# Theory says...
# identity = beta_0 (both costs 0)
# neutral = beta_0 + beta_1 * computation cost = beta_0 + beta_1 * log(1/.9)
# unrelated = beta_0 + beta_1 * computation cost = beta_0 + beta_1 * log(1/.1)
# semantic = beta_0 + beta_1 * computation cost + beta_2 * control cost

rp$s = 0
rp$h = 0
rp[rp$condition == "identity",]$s = basic_sim[basic_sim$goal == "name" & basic_sim$d == 0,]$s
rp[rp$condition == "identity",]$h = basic_sim[basic_sim$goal == "name" & basic_sim$d == 0,]$h
rp[rp$condition == "semantic",]$s = basic_sim[basic_sim$goal == "name" & basic_sim$d == 1,]$s
rp[rp$condition == "semantic",]$h = basic_sim[basic_sim$goal == "name" & basic_sim$d == 1,]$h
rp[rp$condition == "neutral",]$s = neutral_sim[neutral_sim$goal == "name" & neutral_sim$d == 2,]$s
rp[rp$condition == "neutral",]$h =  neutral_sim[neutral_sim$goal == "name" & neutral_sim$d == 2,]$h
rp[rp$condition == "unrelated",]$s = basic_sim[basic_sim$goal == "name" & basic_sim$d == 9,]$s
rp[rp$condition == "unrelated",]$h = basic_sim[basic_sim$goal == "name" & basic_sim$d == 9,]$h

# s = 0.179236
# h = 5.615224e-04

rp = rp %>%
  mutate(condition=if_else(condition == "identity", "congruent", if_else(condition == "semantic", "incongruent", condition))) %>%
  mutate(condition=factor(condition, levels=c("congruent", "neutral", "incongruent", "unrelated"))) 
  
rp %>%
  select(condition, s, h) %>%
  distinct() %>%
  gather(key, value, s, h) %>%
  mutate(key=if_else(key == "s", "Computation cost", "Decision cost")) %>%
  ggplot(aes(x=condition, y=value/log(2), color=condition)) +
    geom_point(shape="triangle", size=5) +
    facet_wrap(~key, scale="free_y") +
    theme_minimal() +
    ylab("Cost (bits)") +
    xlab("") +
    theme(legend.position="bottom", legend.title=element_blank())

ggsave("neutral_simulation.pdf", width=6, height=4)
  

mean_m = lm(mean ~ h + s, data=filter(rp, measure == "M"))
mu_m = lm(mean ~ h + s, data=filter(rp, measure == "mu"))
tau_m = lm(mean ~ h + s, data=filter(rp, measure == "tau"))


prediction_plot = function(rp, the_measure, s_weight, h_weight, bias) {
  to_plot = rp %>% 
    filter(measure == the_measure) %>%
    mutate(predicted_mean=s_weight*s + h_weight*h + bias) %>%
    gather(key, value, mean, predicted_mean) %>% 
    mutate(key=if_else(key == "mean", "empirical", "model")) %>%
    mutate(interval=1.96*se) %>%
    mutate(interval=if_else(key == "model", 0, interval)) %>%
    mutate(interval=na_if(interval, 0))
    
  to_plot %>%
    ggplot(aes(x=condition, y=value, ymin=value-interval, ymax=value+interval, color=condition, shape=key)) +
      geom_point(size=5) +
      geom_errorbar() +
      facet_wrap(~key) +
      theme_bw() +
      theme(legend.title=element_blank(), legend.position="bottom")
}

prediction_plot(rp, "M", 30/log(2), 50/log(2), 725) + ylab("Mean RT (ms)")
ggsave("mean_rp_simulation.pdf", width=6, height=4)

prediction_plot(rp, "mu", 25/log(2), 25/log(2), 610) + ylab("mu (ms)")
ggsave("mu_rp_simulation.pdf", width=6, height=4)

prediction_plot(rp, "tau", 0, 40/log(2), 120) + ylab("tau (ms)")
ggsave("tau_rp_simulation.pdf", width=6, height=4)
