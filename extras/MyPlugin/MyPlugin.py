# -*- coding: utf-8 -*-
'''Elipse Plant Manager - EPM Dataset Analysis Plugin - Plugin example
Copyright (C) 2019 Elipse Software.
Distributed under the MIT License.
(See accompanying file LICENSE.txt or copy at http://opensource.org/licenses/MIT)
'''

# EPM Plugin module
import Plugins as ep
import numpy as np

@ep.DatasetFunctionPlugin('My MAvg filter', 1, 'yf')
def myMAvgPlugin():
    """ Moving average filter (3th order).
    """

    if len(ep.EpmDatasetPens.SelectedPens) != 1:
        ep.showMsgBox('EPM Python Plugin - example', 'Please select a single pen before applying this function!', 'Warning')
        return 0
    order = 3
    epmData = ep.EpmDatasetPens.SelectedPens[0].Values
    yf = filtMAvg(epmData, order)
    ep.plotValues('yf', yf)
    return yf


def filtMAvg(epmData, order):
    yf = epmData.copy()
    yf['Value'][:order] = epmData['Value'][:order].mean()
    for i in range(order + 1, len(epmData) + 1):
        yf["Value"][i-1] = epmData["Value"][i-order:i].mean()
    return yf
