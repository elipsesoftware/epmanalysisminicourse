# -*- coding: utf-8 -*-
"""Elipse Plant Manager - EPM Analysis Minicourse - examples
Copyright (C) 2019 Elipse Software.
Distributed under the MIT License.
(See accompanying file LICENSE.txt or copy at http://opensource.org/licenses/MIT)


ATENÇÃO
=======
Os exemplos neste arquivo servem apenas como referência para o aprendizado, NÃO DEVEM SER UTILIZADOS EM PRODUÇÃO, ou seja, não são feitas verificações de entradas por parte dos usuários, não há tratamento de exceções, não tem testes unitários, nem documentação/help, etc.
Em resumo, NÃO TEM UM MÍNIMO DE GARANTIAS para uso em casos reais!!!

NOTAS
=====

* Para o desenvolvimento de códigos em linguagem Python é altamente recomendado estudar e aplicar as boas práticas definidas na PEP 8 -- Style Guide for Python Code (https://www.python.org/dev/peps/pep-0008/).

* As PEPs são propostas de melhorias da linguagem, PEP 0 -- Index of Python Enhancement Proposals (https://www.python.org/dev/peps/), e a PEP-8 é muito difundida e seguida entre a comunidade que procura, na medida do possível aplicar suas orientações a fim de produzir códigos dentro de um "padrão de estilo" que facilita o desenvolvimento colaborativo.

* Para a criação de módulos (bibliotecas de funções) é altamente recomendado que se pesquise pela PEP 484 -- Type Hints (https://www.python.org/dev/peps/pep-0484/), principalmente quando for desenvolver aplicações ("soluções") para uso de terceiros e/ou mesmo para facilitar a manutenção e desenvolvimento compartilhado do código pela própria equipe.

* Para poder carregar a "my_module", é preciso que a mesma esteja na lista de "paths" da área de trabalho Python, caso contrário, deve-se adicioná-la através dos seguintes comandos:
 >> sys.path.append(r'C:\MyLibs')
 >> import my_module as mm

* Para recarregar um módulo (sem a necessidade de reabrir uma aba do EPM Dataset Analysis, por exemplo):
>> import imp
>> imp.reload(mm)
"""

print('*** Importacao do modulo my_module - Minicurso EPM-Analysis ***')
import numpy as np
from scipy import optimize
from scipy import interpolate
from scipy import integrate
from scipy import signal
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
import csv

def parabola( x, a=1, b=-5, c=6, plotResult=True ):
    """ Função que retorna o resultado da avaliação de uma equação do tipo:
    a*x**2 + b*x + c
    ex.:
    x = np.arange(-10, 11)
    y = parabola(x, 1, 3, 4)
    """
    y = a * x**2 + b * x + c
    if plotResult:
        plt.plot(x, y)
        plt.show()
    return y

# Filtro de média móvel valor repetido no início - informa objeto de dados do EPM
def filtroMM( datasetPen, windowSize ):
    """
    datasetPen = Pena Dataset Analysis.
    windowSize = número de valores para janela de soma.
    Retorna o mesmo array de entrada, porém com os valores de media móvel calculados.
    Obs.: valores repetidos no inicio
    """
    filtrado = datasetPen.copy()
    filtrado["Value"][:windowSize] = datasetPen["Value"][:windowSize].mean()
    for i in range(windowSize+1, len(datasetPen)+1):
        filtrado["Value"][i-1] = datasetPen["Value"][i-windowSize:i].mean()
    return filtrado

# Filtro de media móvel de ordem "o" - informa vetor do numpy
def movingAvgFilterNpVector( xo, o ):
    x = xo.copy() # copia os dados para não alterar a variável do console
    X = x.copy()
    for i in range( 1, o ):
        tmp = np.hstack( (x[i:], np.zeros(i)) )
        X += tmp
    X /= o
    # Repete a ultima media para os valores finais
    lastM = X[-o]
    for i in range( 1, o ):
        X[-i] = lastM
    return X

# Filtro de media móvel de ordem "o" - informa objeto de dados do EPM
def movingAvgFilter( xepmo, o ):
	xepm = xepmo.copy() # copia os dados para nao alterar a vairavel do console
	xf = movingAvgFilterNpVector( xepm['Value'], o )
	xepm['Value'] = xf.copy()
	return xepm

# Filtro de ruído de um sinal utiliza Butterworth ordem 2 - informa objeto de dados do EPM
def filt_signal( xepmo, w = 0.3, o = 2 ):
    # Parametros do filtro baseado em Butterworth
    b, a = signal.butter( o, w )
    yfVec = filt_filt( b, a, xepmo['Value'] )
    yf = xepmo.copy()
    yf['Value'] = yfVec
    return yf

# Exportar para arquivo CSV - usando o módulo numpy
def export_csv( epmData, fileName ):
	np.savetxt( fileName, epmData['Value'], fmt='%0.0f', delimiter=';')

# Importar de um arquivo CSV (apenas duas colunas de dados)
def read_from_csv( fileName, delimiter=';' ):
    f = csv.reader(open(fileName), delimiter=delimiter)
    cd1, cd2 = [],[]
    for (c1, c2) in f:
        c1 = c1.replace(',', '.')
        c2 = c2.replace(',', '.')
        cd1.append(float(c1))
        cd2.append(float(c2))
    return np.array(cd1), np.array(cd2)

# Funçãoo que conta o numero de inversões de sentido
def invCount( a ):
    d = a[1:] - a[:-1] # delta
    n = 0
    for i in range(len(d)-1):
        if np.sign(d[i]) != np.sign(d[i+1]):
           n += 1
    return n

# Equação da reta
def eqReta(x, y):
	n = len(x)
	xy = x*y
	xm = x.mean()
	ym = y.mean()
	x2 = x**2
	a = ( xy.sum() -  n * xm * ym ) / ( x2.sum() - n* xm**2)
	b = ym - a * xm
	r = np.corrcoef([x,y])[1][0] # coeficiente de correlação
	return a,b,r

# Retorna as dimensões de uma matriz
def matSize( A ):
    if (len(A) == np.size(A)):
        return len(A), 1 # interpreta o vetor como sendo de apenas uma coluna
    else:
        return int(np.shape(A)[0]), int(np.shape(A)[1])

# Função para geração de um sinal APRBS informando os períodos mínimo e máximo de permanência entre os movimentos.
def aprbs( n, pmin, pmax, opt = 0, F = 0.8 ):
# n     numero de pontos a serem gerados
# pmin  patamar mínimo (inteiro >= 1)
# pmax  patamar máximo (inteiro >= 1)
# opt   opt = 1 sinal com diferentes intensidades
# F     parâmetro de distribuição (0.001 <= F <= 0.999).
# F = 1 significa distribuição exponencial entre trocas de patamares
# F = 0 significa distribuição uniforme entre trocas de patamares
	if( pmin < 1 ):
		pmin = 1
	if( pmax < pmin):
		pmax = pmin
	if( n < 1 ):
		n = 1
	miny = pmin - 0.4999
	maxy = pmax + 0.4999
	maxu = 1
	minu = (1 - F) * maxu
	deltu = maxu - minu
	delta = (maxy - miny) / np.log(minu)
	state = int(np.sign( np.random.rand(1) - 0.5))
	if( state == 0 ):
		state = 1
	j = 0
	u = np.zeros(n)
	while j <= n-1:
		t  = int(round(( miny + np.log(minu + np.random.rand(1) * deltu) * delta)[0]))
		state = -state
		if opt:
			aux = state*np.random.rand(1)
		else:
			aux = state
		for k in range(t):
			u[j] = aux
			j = j + 1
			if j > n-1:
				break
	return u

# Estimativa grosseira da amplitude do ruido de um sinal
def amplitudeEstimation( x, w = 0.5 ):
    # Parâmetros do filtro baseado em Butterworth
    o = 2 # ordem
    b, a = signal.butter( o, w )
    # Filtra o sinal
    xf = filt_filt( b, a, x)
    # Calcula o a diferença pto a pto entre sinal original e o filtrado
    dx = abs( x - xf )
    # Estima a amplitude baseada na amplitude media e o desvio padrão (2 desvios)
    amp = dx.mean() + 2 * dx.std()
    ## Plota os resultados
    plt.plot(x,'c')
    plt.plot(xf,'k')
    plt.legend(('original','filtrado'))
    plt.show()
    return amp

# Gráfico de histograma informando dados do EPM
def histPlot( xepm, ei = -1, es = -1 ):
    nd = len(xepm) # numero de pontos
    ni = int(np.sqrt(nd)) # numero de intervalos
    xo = xepm['Value'].copy()
    mu = xo.mean() # media da distribuição
    sigma = xo.std() # desvio padrão
    if ei < 0:
       ei = xo.min()
    if es < 0:
       es = xo.max()
    # Capacidade de produzir dentro das especificações e reprodutibilidade nas medidas (processo centrado nas especificações de Engenharia)
    # Cp = (EspecSup - EspecInfer)/(6 sigma )                     ISO TS16949 (manufatura) Cp > 1,33
    # Cpk = Min [ (Media-EspInf)/3Sigma; (EspSup-Media)/3Sigma ]  ISO TS16949 (manufatura) Cpk > 1,67
    cp  = (es-ei)/(6*sigma)
    cpk = min([ (mu-ei)/(3*sigma) , (es-mu)/(3*sigma)])
    n, bins, patches = plt.hist(xo, ni, facecolor='green', alpha=0.5)
    y = stats.norm.pdf(bins, mu, sigma) # envelope da curva normal
    plt.plot(bins, y, 'r--')
    title = r'Histograma: $\mu='
    title += str(mu)
    title +=r'100$, $\sigma='
    title += str(sigma)
    title += '$'
    plt.title(title)
    font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'bold',
        'size'   : 11,
        }
    CpCpk = r'Cp = '
    CpCpk += str(cp)
    CpCpk += ' - ISO TS16949: Cp > 1.33\n'
    CpCpk += 'Cpk = '
    CpCpk += str(cpk)
    CpCpk += ' - ISO TS16949: Cpk > 1.67'
    dx = plt.axis()[1]-plt.axis()[0]
    dy = plt.axis()[3]-plt.axis()[2]
    xpos = plt.axis()[0] + dx/10
    ypos = plt.axis()[3] - dy/10
    plt.text(xpos, ypos, CpCpk, fontdict=font)
    plt.show()

# Gráfico com infos de CEP
def cep( xepm, ei = None, es = None ):
    # \TODO: melhorar apresentação das informações nos gráficos (gerar HTML?!)
    n = len(xepm) # numero de pontos
    x = xepm['Value'].copy()
    t = xepm['Timestamp'].copy()
    mu = x.mean() # media da distribuição
    sigma = x.std() # desvio padrão
    if ei == None:
       ei = x.min()
    if es == None:
       es = x.max()
    # Cálculo do Quartil_1 (25%), Quartil_3 (75%) e IQR = Quartil_3 - Quartil_1
    q1 = stats.scoreatpercentile(x,25)
    q3 = stats.scoreatpercentile(x,75)
    iqr = q3 - q1
    # Cálculo do tamanho dos bins - regra de Freedman-Diaconis: binSize = 2 * IQR * n**(-1/3)  , onde n é o tamanho da amostra
    binsSize = 2.0 * iqr * 1 / n**(1/3.)
    binsNum = int( (x.max()- x.min())/binsSize ) + 1
    # Mostra o histograma dos dados, o envelope com a curva normal e o Box Plot
    pOut = 1.5
    fig = plt.figure(1, figsize=(10.5,10.5))
    axHist = fig.add_subplot(5, 1, 1)
    m, bins, patches = axHist.hist(x, binsNum, facecolor='green', alpha=0.5)
    axHist.set_aspect(1.)
    y = stats.norm.pdf(bins, x.mean(), x.std())  # envelope da curva normal
    plt.plot(bins, y, 'r--')
    divider = make_axes_locatable(axHist)
    axBoxplot = divider.append_axes("bottom", 1.2, pad=0.1, sharex=axHist)
    plt.setp(axHist.get_xticklabels(), visible=False)
    axBoxplot.boxplot(x, sym='ro', vert=False, whis=pOut)
    # *** Cálculo das demais informações estatisticas ***
    # Remoção dos dados referentes as causas especiais (outliers) -  less than (Q1 - pOut * IQR)  OR  greater than (Q3 + pOut + IQR)
    outMin = q1 - pOut*iqr # valores menores que outMin serão considerados outliers
    outMax = q3 + pOut*iqr # valores maiores que outMax serão considerados outliers
    vecOut = (x > outMin) * (x < outMax) # vetor com posições outliers = False
    posOut = np.argwhere(vecOut).ravel() # vetor com os índices das posições com dados validos (não são causas especiais)
    x1 = x[vecOut] # dados sem os pontos outliers
    t1 = t[vecOut] # timestamp dos dados sem os pontos outliers
    # Q-Qplot para avaliar suposição de distribuição normal
    axQQPlot = fig.add_subplot(5, 1, 2)
    tQs, eq = stats.probplot(x1, dist="norm", plot=plt)
    # Calcula as estatísticas sobre os dados SEM as causas especiais
    size, minmax, amean, var, skewness, kurtosis = stats.describe(x1)
    # Teste Shapiro-Wilk (avaliar a hipótese de distribuição normal)
    w, p = stats.shapiro(x1)
    # Calculo dos limites de controle (valores)
    sec2min = np.vectorize(lambda x: int(x/60. + 0.5))
    td2sec = np.vectorize(lambda dt: dt.total_seconds())
    if len(x) == len(x1):   # outra forma de verificar seria:  np.any(posOut[1:]-posOut[:-1] >1) -> Se for verdadeiro eh pq dados foram removidos
        mr = abs(x1[1:] - x1[:-1]) # supondo que não foram removidos outliers no interior de x1
        mrt = td2sec(t1[1:] - t1[:-1])
        mrt = mrt.cumsum() # delta em segundos
    else:
        mr = []
        mrt = []
        for i in range(posOut.size - 1):
            if (posOut[i+1] - posOut[i])== 1: # se não foi removido algum valor, a diferença dos índices deve ser igual a 1
               mr.append(abs(x1[i+1] - x1[i]))
               mrt.append((t1[i+1] - t1[i]).total_seconds())
        mr = np.array(mr)
        mrt = np.array(mrt).cumsum() # delta em segundos
    mrt_min = sec2min(mrt) # delta em minutos, com arredondamento
    dt1 = np.zeros(len(x1))
    dt1[1:] = td2sec(t1[1:] - t1[:-1])
    dt1 = dt1.cumsum()
    licV = x1.mean() - 2.66 * mr.mean()
    lscV = x1.mean() + 2.66 * mr.mean()
    # Cálculo dos limites de controle (amplitude)
    licA = 0
    lscA = 3.267 * mr.mean()
    # Cálculo dos índices de desempenho (Pp, Ppk)
    lse = 2.6 # Limite superior especificado pelo usuário
    lie = 2.4 # Limite inferior especificado pelo usuário
    pp = (lse - lie) / (6 * x1.std())
    ppk = min([lse - x1.mean(), x1.mean() - lie]) / (3 * x1.std())
    # Cálculo dos índices de capabilidade (Cp, Cpk)
    mrStd = mr.mean() / 1.128 # desvio padrão estimado a partir das amplitudes
    cp = (lse - lie) / (6 * mrStd)
    cpk = min([lse - x1.mean(), x1.mean() - lie]) / (3 * mrStd)
    # Gráfico Serie temporal dos limites de controle
    axLCVs = fig.add_subplot(5, 1, 3)
    axLCVs.plot(dt1, x1)
    axLCVs.plot(dt1, licV * np.ones(len(x1)), 'r--')
    axLCVs.plot(dt1, lscV * np.ones(len(x1)), 'r--')
    axLCAs = fig.add_subplot(5, 1, 4)
    axLCAs.plot(mrt, mr)
    axLCAs.plot(mrt, licA * np.ones(len(mr)), 'r--')
    axLCAs.plot(mrt, lscA * np.ones(len(mr)), 'r--')
    # Apresentação dos resultados
    font = {'family' : 'serif', 'color'  : 'darkred','size'   : 9}
    axInfos = fig.add_subplot(5, 1, 5)
    plt.setp(axInfos.get_xticklabels(), visible=False)
    plt.setp(axInfos.get_yticklabels(), visible=False)
    xText = 0.01
    yText = 0.95
    yOffset = 0.08
    axInfos.text(xText, yText, 'size of the data: ' + str(size), fontdict=font)
    axInfos.text(xText, yText - 1 * yOffset, 'minimum: ' + str(minmax[0]), fontdict=font)
    axInfos.text(xText, yText - 2 * yOffset, 'maximum: ' + str(minmax[1]), fontdict=font)
    axInfos.text(xText, yText - 3 * yOffset, 'arithmetic mean: ' + str(amean), fontdict=font)
    axInfos.text(xText, yText - 4 * yOffset, 'unbiased variance: ' + str(var), fontdict=font)
    axInfos.text(xText, yText - 5 * yOffset, 'biased skewness: ' + str(skewness), fontdict=font)
    axInfos.text(xText, yText - 6 * yOffset, 'biased kurtosis: ' + str(kurtosis), fontdict=font)
    axInfos.text(xText, yText - 7 * yOffset, 'W (Shapiro-Wilk test): ' + str(w), fontdict=font)
    axInfos.text(xText, yText - 8 * yOffset, 'p (Shapiro-Wilk test): ' + str(p), fontdict=font)
    axInfos.text(xText, yText - 9 * yOffset, 'LIC (valor): ' + str(licV), fontdict=font)
    axInfos.text(xText, yText - 10 * yOffset, 'LSC (valor): ' + str(lscV), fontdict=font)
    axInfos.text(xText, yText - 11 * yOffset, 'LIC (amplitude): ' + str(licA), fontdict=font)
    axInfos.text(xText, yText - 12 * yOffset, 'LSC (amplitude): ' + str(lscA), fontdict=font)
    axInfos.text(xText, yText - 13 * yOffset, 'Pp: ' + str(pp), fontdict=font)
    axInfos.text(xText, yText - 14 * yOffset, 'Ppk: ' + str(ppk), fontdict=font)
    axInfos.text(xText, yText - 15 * yOffset, 'Cp: ' + str(cp), fontdict=font)
    axInfos.text(xText, yText - 16 * yOffset, 'Cpk: ' + str(cpk), fontdict=font)
    plt.show()


# Função que envia um e-mail com uma imagem anexada
def sendMailTo(to_addr, subject_header, bd='', att=''):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.MIMEText import MIMEText
    from email.mime.image import MIMEImage
    from_addr = 'SERVER.Empresa.com.br' # \TODO: informar um Servidor valido
    mail_username = 'user@Empresa.com.br' # \TODO: informar um Usuario valido
    mail_password = 'PASSWORD' # \TODO: informar uma Senha valida
    mail_server = 'smtp.Epmresa.com.br' # \TODO: informar um SMTP valido
    mail_server_port = 999 # \TODO: informar uma porta valida
    attachment = att
    body = bd
    msg = MIMEMultipart()
    msg["To"] = to_addr
    msg["From"] = from_addr
    msg["Subject"] = subject_header
    msgText = MIMEText('<b>%s</b>' % body, 'html')
    msg.attach(msgText)
    if len(attachment):
       fp = open(attachment, 'rb')
       img = MIMEImage(fp.read())
       fp.close()
       msg.attach(img)
    email_message = '%s\n%s' % (subject_header, body)
    smtpserver = smtplib.SMTP(mail_server, mail_server_port)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(mail_username, mail_password)
    smtpserver.sendmail(from_addr, to_addr, msg.as_string())
    smtpserver.quit()

# Retorna o percentual de tempo que a variável ficou em cada período
def percentTimeIn(epmData, nodes = -1):
    t,y = rmNanAndOutliers2(epmData)
    # Se não foi informado os intervalos, ele cria automaticamente 3 intervalos
    if nodes == -1:
       minVal = int(np.floor(np.nanmin(y))) # não precisaria usar a função nanmin, pois os Nan já foram removidos
       maxVal = int(np.ceil(np.nanmax(y))) # não precisaria usar a função nanmax, pois os Nan já foram removidos
       step = int((maxVal-minVal)/3)
       nodes = range(minVal,maxVal,step)
    intervNum = np.size(nodes)+1 # [antes - intervalos - depois] (nodes padrão utilizou o min/max, logo não tera valores fora deste intervalo)
    totTime = np.empty(intervNum, dtype=object) # cria um vetor vazio com a dimensão do num de intervalo mais antes e depois
    totTime.fill(datetime.timedelta(0,0)) # preenche o vetor com tipo timedelta em 0
    for i in range(1,np.size(y)):
        dt = t[i] - t[i-1]
        ix = np.digitize([y[i]], nodes)
        totTime[ix] += dt
    nodesPercents = np.zeros([np.size(totTime),2]) # cria uma matriz vazia - col1: intervalo  col2:valor percentual
    totalPeriod = totTime.sum().total_seconds()
    for i in range(np.size(totTime)):
        if i:
           nodesPercents[i,0] = nodes[i-1]
        else:
           nodesPercents[i,0] = -np.inf
        nodesPercents[i,1] = totTime[i].total_seconds()/totalPeriod
    labels = []
    for item in nodesPercents[:,0]:
        labels.append(str(item))
    # pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.pie(nodesPercents[:,1], labels=labels, autopct='%1.1f%%', shadow=True)
    plt.show()
    return nodesPercents

# Remove dados Nan e Outliers baseado no desvio padrão
def rmNanAndOutliers(epmData, sd = 6):
    y = epmData['Value']
    t = epmData['Timestamp']
    # Remove os valores Nan de y e os correspondentes timestamps
    nanPos = np.argwhere(np.isnan(y)) # retorna os indices da matriz que sao Nan
    y = np.delete(y,nanPos) # deleta as os valores das posicoes que contem nan
    t = np.delete(t,nanPos)
    # Remove pontos que estao 4 desvios alem da media
    s3 = np.floor(sd * y.std())
    smin = y.mean() - s3
    smax = y.mean() + s3
    outPos = np.argwhere(y<smin) # retorna os indices da matriz que sao menores que smin
    y = np.delete(y,outPos) # deleta as os valores das posicoes outPos
    t = np.delete(t,outPos)
    outPos = np.argwhere(y>smax) # retorna os indices da matriz que sao maiores que smax
    y = np.delete(y,outPos) # deleta as os valores das posicoes outPos
    t = np.delete(t,outPos)
    res = vec2epm(t,y)
    return res

# Remove dados Nan e Outliers baseado no desvio padrao e retorna vetores t e y
def rmNanAndOutliers2(epmData, sd = 6):
    y = epmData['Value']
    t = epmData['Timestamp']
    # Remove os valores Nan de y e os correspondetes timestamps
    nanPos = np.argwhere(np.isnan(y)) # retorna os indices da matriz que sao Nan
    y = np.delete(y,nanPos) # deleta as os valores das posicoes que contem nan
    t = np.delete(t,nanPos)
    # Remove pontos que estao 4 desvios alem da media
    s3 = np.floor(sd * np.sqrt(y.std()))
    smin = y.mean() - s3
    smax = y.mean() + s3
    outPos = np.argwhere(y<smin) # retorna os indices da matriz que sao menores que smin
    y = np.delete(y,outPos) # deleta as os valores das posicoes outPos
    t = np.delete(t,outPos)
    outPos = np.argwhere(y>smax) # retorna os indices da matriz que sao maiores que smax
    y = np.delete(y,outPos) # deleta as os valores das posicoes outPos
    t = np.delete(t,outPos)
    return t,y

# Retorna um objeto de dados a partir de um vetor de tempo e outro de dados, as qualidades sao sempre boas
def vec2epm(t, y):
    """ Parametros
    t: vetor com os tempos
    y: vetor com os dados
    """
    desc = np.dtype([('Value', '>f8'), ('Timestamp', 'object'), ('Quality', 'object')])
    epmY = np.empty([np.size(y)], dtype = desc)
    epmY['Value'] = y
    epmY['Timestamp'] = t
    epmY['Quality'] = 0
    return epmY

# Geração do perfil diario de uma variavel - retorna uma matriz para plotar com matplotlib.pylab.plot (dados devem fechar em dia completo)
# Os dados devem estar interpolados (igualmente espacados).
def dailyProfile3D( epmData, sampling = 30, pHours = 24):
    """
    epmData: dados do EPM
    sampling: amostragem da interpolacao - informada em minutos
    pHours: intervalo para geracao do perfil
    """
    iniPeriod = epmData['Timestamp'][0]
    endPeriod = epmData['Timestamp'][-1]
    dTotal = endPeriod - iniPeriod
    totDays = dTotal.days # numero de dias a serem avaliados
    evalPeriod = datetime.timedelta(hours = pHours) # periodo de avaliacao - busca dos valores minimos
    nextPeriod = iniPeriod + evalPeriod
    # Pesquisa diaria segundo base horaria definida em iniPeriod ate evalPeriod (horas)
    profileList = []
    for i in range(totDays+1):
        iniP = epmData['Timestamp'] >= iniPeriod
        endP = epmData['Timestamp'] < nextPeriod
        epmDataValue = epmData['Value']
        dataPeriod = epmDataValue[iniP * endP]
        profileList.append( dataPeriod )
        # Atualiza os periodos para o proximo dia
        iniPeriod = iniPeriod + datetime.timedelta( 1 )
        nextPeriod = iniPeriod + evalPeriod
    profileMatrix = np.array(profileList)
    days  = np.arange(totDays)+1
    hours = np.arange(0,pHours*60,sampling)
    meshTime, indices = np.meshgrid(hours, days)
    meshProfile = np.zeros(meshTime.shape)
    for i in range( indices.shape[0] ):
        for j in range( indices.shape[1] ):
            meshProfile[i,j] = profileMatrix[i,j]
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = meshTime
    Y = indices
    Z = meshProfile
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm', alpha=0.8)
    ax.set_xlabel('minutos')
    ax.set_ylabel('dia')
    plt.show()


##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

# Curva de potência com o vento - informando a P nominal
# Pot = Pnominal / (1 + exp-(a * V + b))
# Curva de potência com o vento - estimando a P nominal
def powerFitPn(par, x):
    return par[2] / (par[3] + np.exp(-(par[0] * x + par[1])))

# Resíduos para estimar pârametros da curva de Pot com o Vento estimando tbm Pnominal
def residualsSPPn(par, x, y):
    return powerFitPn(par, x) - y

# Analise de dados Energia Eólica - Wind Power Generator - verso com Spline
def windPower(epmWindSpeed, epmPower, Pnominal):
    speedData = epmWindSpeed['Value'].copy()
    powerData = epmPower['Value'].copy()
    # Removendo Valores de Potências igual a zero
    pPos = np.argwhere(powerData < 0.)
    speed = np.delete(speedData,pPos)
    power = np.delete(powerData,pPos)
    # Remove potências superiores a potência nominal informada
    pPos = np.argwhere(powerData > Pnominal)
    speed = np.delete(speed,pPos)
    power = np.delete(power,pPos)
    # Remove Valores inferiores a velocidades de vento inferiores a 4 m/s
    pPos = np.argwhere(speed < 4.)
    speed = np.delete(speed,pPos)
    power = np.delete(power,pPos)
    # Grafico scatter da potência com a velocidade do vento
    plt.scatter(speed, power)
    # Cálculo das potências medias para cada velocidade
    xm, ym = windPowerAverage(speed, power)
    # Dados para fazer a curva de referência via arquivo CVS
    speedRef, powerRef = read_from_csv(r'C:\MyLibs\Refdata2000.csv')
    # Fit via Spline para plotar a curva de referência
    tckRef = interpolate.splrep(speedRef, powerRef, s=0)
    binSpeed = 0.5
    xRef = np.arange(speedRef.min(), speedRef.max(), binSpeed)
    yRef = interpolate.splev(xRef, tckRef, der=0)
    # Dados da velocidade igualmente espaçados para plotar a curva estimada
    # Fit via Eq com 4 parâmetros estimados
    par0 = [1.0, 1.0, 1500.0, 1.0]
    parest,cov,infodict,mesg,ier = optimize.leastsq(residualsSPPn, par0, args=(xm, ym), full_output=True)
    xEst = np.arange(xm.min(), xm.max(), binSpeed)
    yEst = powerFitPn(parest, xEst)
    posAbove = np.argwhere(yEst > Pnominal) # verifica se algum valor da spline deu superior a Pnominal e aplica clamping
    yEst[posAbove] = Pnominal
    # Cálculo da área entre as curvas de refer~ncia e a obtida através dos dados medidos -> Energia que não esta sendo gerada
    energyLost = integrate.simps(yRef, dx=binSpeed) - integrate.simps(yEst, dx=binSpeed)
    # Plot da curva de referência
    plt.plot(xRef, yRef, color='r', linewidth=3)
    # Plot da curva estimada a partir dos dados
    plt.plot(xEst, yEst, color='g', linewidth=2)
    plt.text(15, 1000, str(round(energyLost/1000,2)) + '(MW)', fontsize=12)
    plt.xlabel(r'$Speed (m/s)$')
    plt.ylabel(r'$Power (KW)$')
    plt.show()

# Determinação dos valores médios de potência para cada valor de velocidade
def windPowerAverage(speed, power):
    pos = np.argsort(speed)
    x = speed[pos].copy()
    y = power[pos].copy()
    xm = []
    ym = []
    i = 0
    while i < (len(x)):
        p = np.where(x == x[i])
        xm.append(x[p].mean())
        ym.append(y[p].mean())
        i = p[0][-1] + 1
    return np.array(xm), np.array(ym)
#####################################################
#####################################################
# Funções extras para filtro de sinais
def lfilter_zi( b, a ):
    n = max(len(a),len(b))
    zin = (  np.eye(n-1) - np.hstack( (-a[1:n, np.newaxis], np.vstack((np.eye(n-2), np.zeros(n-2))))))
    zid=  b[1:n] - a[1:n]*b[0]
    zi_matrix=np.linalg.inv(zin)*(np.matrix(zid).transpose())
    zi_return=[]
    for i in range(len(zi_matrix)):
      zi_return.append(float(zi_matrix[i][0]))
    return np.array( zi_return )

def filt_filt( b, a, x ):
    # Filtro nos dois sentidos - apenas para vetores
    ntaps = max(len(a),len(b))
    edge = ntaps * 3
    if x.ndim != 1:
        print("Apenas arrays de dimensao 1.")
    # x deve ser maior que o extremo
    if x.size < edge:
        print("Vetor de entrada maior que 3 * max(len(a),len(b).")
    if len(a) < ntaps:
        a = np.r_[a,np.zeros(len(b)-len(a))]
    if len(b) < ntaps:
        b = np.r_[b,np.zeros(len(a)-len(b))]
    zi = lfilter_zi( b, a )
    # Amplia o sinal para ter os extremos ao aplicar o sinal no sentido inverso
    s = np.r_[2*x[0]-x[edge:1:-1],x,2*x[-1]-x[-1:-edge:-1]]
    # Filtrando no sentido direto
    ( y, zf ) = signal.lfilter( b, a, s, -1, zi*s[0] )
    # Filtrando no sentido inverso - removendo a fase
    ( y, zf ) = signal.lfilter( b, a, np.flipud(y), -1, zi*y[-1] )
    # Removendo os extremos adicionados
    return np.flipud( y[edge-1:-edge+1] )

