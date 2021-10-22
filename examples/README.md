## IFdensity Examples 

The following GIFs show the influence functions of the logarithm of the score matching 
density estimate in a kernel exponential family under various scenarios.

This notebook looks at the influence function of the logarithm of the score matching density function
in a kernel exponential family $\mathcal{Q}$ evaluated at a point $w$, which is defined as 

```math
\mathrm{IF} \big( y; \log q (w; F) \big) := \lim_{\varepsilon \to 0^+} \frac{1}{\varepsilon} \Big(\log q \big(w; (1 - \varepsilon) F + \varepsilon \delta_y \big) - \log q \big(w; F\big)\Big), \quad \text{ for all } w \in \mathcal{X}, \hspace{50pt} (*)
```

where $\mathcal{X} \subseteq \mathbb{R}$ is the sample space, $F$ is a probability distribution over $\mathcal{X}$, $q (\cdot; F): \mathcal{X} \to [0, \infty)$ is the score matching density function in $\mathcal{Q}$ under $F$, $\varepsilon \in (0, 1]$, and $\delta_y$ is the point mass 1 at $y \in \mathcal{X}$. 

We approximate $(*)$ by 

$$
\widehat{\mathrm{IF}} \big( y; \log q (w; F_n) \big) := \frac{1}{\varepsilon} \Big(\log q \big(w; (1 - \varepsilon) F_n + \varepsilon \delta_y\big) - \log q \big(w; F_n\big)\Big), \quad \text{ for all } w \in \mathcal{X}, \hspace{50pt} (**)
$$

with a small $\varepsilon$, where $F_n$ is the empirial distribution. 

In the below, we use the `waiting` variable in the Old Faithful Geyser dataset and insert an additional observation, i.e., $y$ in $(**)$, each time. These additional observations are $90$, $92$, $\cdots$, $398$, $400$. In additional, we choose the sample space $\mathcal{X} = (0, \infty)$, the kernel function to be the Gaussian kernel function, the bandwidth parameter to be $5.0$, $7.0$ and $9.0$, the penalty parameter to be $\exp({-12.0})$, $\exp({-10.0})$ and $\exp({-8.0})$, and $\varepsilon$ in $(*)$ to be `1e-8`. 

**Observation:** When the contaminated observation (labeled by the purple vertical line and red rug) 
is kept being moved to right and is sufficient far away from others (labeled by the blue rugs), 
the resulting influence function becomes very stable in the sense that the next one is essentially 
a shift of the previous one to the right.  

![alt text](gif/IF-logdensity-waiting-kernel=gaussian_poly2-bw=5.0-pen=exp-8.0-contamweight=1e-08.gif)

![alt text](gif/IF-logdensity-waiting-kernel=gaussian_poly2-bw=5.0-pen=exp-10.0-contamweight=1e-08.gif)

![alt text](gif/IF-logdensity-waiting-kernel=gaussian_poly2-bw=5.0-pen=exp-12.0-contamweight=1e-08.gif)

![alt text](gif/IF-logdensity-waiting-kernel=gaussian_poly2-bw=7.0-pen=exp-8.0-contamweight=1e-08.gif)

![alt text](gif/IF-logdensity-waiting-kernel=gaussian_poly2-bw=7.0-pen=exp-10.0-contamweight=1e-08.gif)

![alt text](gif/IF-logdensity-waiting-kernel=gaussian_poly2-bw=7.0-pen=exp-12.0-contamweight=1e-08.gif)

![alt text](gif/IF-logdensity-waiting-kernel=gaussian_poly2-bw=9.0-pen=exp-8.0-contamweight=1e-08.gif)

![alt text](gif/IF-logdensity-waiting-kernel=gaussian_poly2-bw=9.0-pen=exp-10.0-contamweight=1e-08.gif)

![alt text](gif/IF-logdensity-waiting-kernel=gaussian_poly2-bw=9.0-pen=exp-12.0-contamweight=1e-08.gif)
