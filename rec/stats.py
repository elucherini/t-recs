# Import scipy 1.4.1 continuous, multivariate, and discrete distributions
from scipy.stats import alpha, anglit, arcsine, argus, beta, betaprime, bradford, \
                        burr, burr12, cauchy, chi, chi2, cosine, crystalball, dgamma, \
                        dweibull, erlang, expon, exponnorm, exponweib, exponpow, f, \
                        fatiguelife, fisk, foldcauchy, frechet_r, frechet_l, genlogistic, \
                        gennorm, genpareto, genexpon, genextreme, gausshyper, gamma, \
                        gengamma, genhalflogistic, geninvgauss, gilbrat, gompertz, \
                        gumbel_r, gumbel_l, halfcauchy, halflogistic, halfnorm, \
                        halfgennorm, hypsecant, invgamma, invgauss, invweibull, johnsonsb, \
                        johnsonsu, kappa4, kappa3, ksone, kstwobign, laplace, levy, \
                        levy_l, levy_stable, logistic, loggamma, loglaplace, lognorm, \
                        loguniform, lomax, maxwell, mielke, moyal, nakagami, ncx2, ncf, \
                        nct, norm, norminvgauss, pareto, pearson3, powerlaw, powerlognorm, \
                        powernorm, rdist, rayleigh, rice, recipinvgauss, semicircular, \
                        skewnorm, t, trapz, triang, truncexpon, truncnorm, tukeylambda, \
                        uniform, vonmises, vonmises_line, wald, weibull_min, weibull_max, \
                        wrapcauchy, multivariate_normal, matrix_normal, dirichlet, wishart, \
                        invwishart, multinomial, special_ortho_group, ortho_group, \
                        unitary_group, random_correlation, bernoulli, betabinom, binom, \
                        boltzmann, dlaplace, geom, hypergeom, logser, nbinom, planck, \
                        poisson, randint, skellam, zipf, yulesimon

# Map distribution functions to their names
names = {'alpha': alpha, 'anglit': anglit, 'arcsine': arcsine, 'argus': argus, 'beta': beta,
        'betaprime': betaprime, 'bradford': bradford, 'burr': burr, 'burr12': burr12,
        'cauchy': cauchy, 'chi': chi, 'chi2': chi2, 'cosine': cosine,
        'crystalball': crystalball, 'dgamma': dgamma, 'dweibull': dweibull, 'erlang': erlang,
        'expon': expon, 'exponnorm': exponnorm, 'exponweib': exponweib, 'exponpow': exponpow,
        'f': f, 'fatiguelife': fatiguelife, 'fisk': fisk, 'foldcauchy': foldcauchy,
        'frechet_r': frechet_r, 'frechet_l': frechet_l, 'genlogistic': genlogistic,
        'gennorm': gennorm, 'genpareto': genpareto, 'genexpon': genexpon,
        'genextreme': genextreme, 'gausshyper': gausshyper, 'gamma': gamma, 'gengamma': gengamma,
        'genhalflogistic': genhalflogistic, 'geninvgauss': geninvgauss, 'gilbrat': gilbrat,
        'gompertz': gompertz, 'gumbel_r': gumbel_r, 'gumbel_l': gumbel_l, 'halfcauchy': halfcauchy,
        'halflogistic': halflogistic, 'halfnorm': halfnorm, 'halfgennorm': halfgennorm,
        'hypsecant': hypsecant, 'invgamma': invgamma, 'invgauss': invgauss, 'invweibull': invweibull,
        'johnsonsb': johnsonsb, 'johnsonsu': johnsonsu, 'kappa4': kappa4, 'kappa3': kappa3,
        'ksone': ksone, 'kstwobign': kstwobign, 'laplace': laplace, 'levy': levy,
        'levy_l': levy_l, 'levy_stable': levy_stable, 'logistic': logistic, 'loggamma': loggamma,
        'loglaplace': loglaplace, 'lognorm': lognorm, 'loguniform': loguniform, 'lomax': lomax,
        'maxwell': maxwell, 'mielke': mielke, 'moyal': moyal, 'nakagami': nakagami,
        'ncx2': ncx2, 'ncf': ncf, 'nct': nct, 'norm': norm, 'norminvgauss': norminvgauss,
        'pareto': pareto, 'pearson3': pearson3, 'powerlaw': powerlaw, 'powerlognorm': powerlognorm,
        'powernorm': powernorm, 'rdist': rdist, 'rayleigh': rayleigh, 'rice': rice,
        'recipinvgauss': recipinvgauss, 'semicircular': semicircular, 'skewnorm': skewnorm,
        't': t, 'trapz': trapz, 'triang': triang, 'truncexpon': truncexpon, 'truncnorm': truncnorm,
        'tukeylambda': tukeylambda, 'uniform': uniform, 'vonmises': vonmises,
        'vonmises_line': vonmises_line, 'wald': wald, 'weibull_min': weibull_min,
        'weibull_max': weibull_max, 'wrapcauchy': wrapcauchy,
        'multivariate_normal': multivariate_normal, 'matrix_normal': matrix_normal,
        'dirichlet': dirichlet, 'wishart': wishart, 'invwishart': invwishart,
        'multinomial': multinomial, 'special_ortho_group': special_ortho_group,
        'ortho_group': ortho_group, 'unitary_group': unitary_group,
        'random_correlation': random_correlation, 'bernoulli': bernoulli, 'betabinom': betabinom,
        'binom': binom, 'boltzmann': boltzmann, 'dlaplace': dlaplace, 'geom': geom,
        'hypergeom': hypergeom, 'logser': logser, 'nbinom': nbinom, 'planck': planck,
        'poisson': poisson, 'randint': randint, 'skellam': skellam, 'zipf': zipf,
        'yulesimon': yulesimon}

'''
'' Wrapper around scipy 1.4.1 distributions
''' 
class Distribution:
    '''
    '' @distr_type: str or callable of scipy distribution
    '' @non_negative: bool
    '' @**kwargs: distribution parameters
    '''
    def __init__(self, distr_type='norm', non_negative=False, **kwargs):
        self.parameters = kwargs
        self.non_negative = non_negative
        if isinstance(distr_type, str) and distr_type in names:
            self.type = distr_type
            distr_f = names[distr_type]#list(names.keys())[list(names.values()).index(distr_type)]
        else:
            raise ValueError("Distribution '%s' does not exist: wrong object name" % str(distr_type))
        self.function = distr_f.rvs

    def compute(self, **params):
        try:
            self.set_parameters(**params)
        except:
            raise
        else:
            result = self.function(**self.parameters)
            if self.non_negative:
                result = abs(result)
            return result


    '''
    '' Add/replace distribution parameters
    '' If params contains a key that is already in self.parameters, it replaces the old
    '' value with the new specified value.
    '''
    def set_parameters(self, **params):
        if params is None:
            raise ValueError("params can't be None")
        p = {**self.parameters, **params}
        try:
            # check that parameters make sense with given distribution
            d = self.function(**p)
        except:
            raise
        else:
            self.parameters = p

    def get_parameters(self):
        return self.parameters

    def get_distribution_type(self):
        return self.type
        

# Test
if __name__ == '__main__':
    params = {'size': (10, 10)}
    p2 = {'size': (3,5)}
    try:
        d = Distribution(distr_type='norm', **params)
        print(d.compute())
        print(d.function)
        print(d.parameters)
        print(d.type)
        print(d.compute(**p2))
        print(d.parameters)
    except Exception as e:
        print(e)

