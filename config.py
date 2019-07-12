# -*- coding: utf-8 -*-

base_model='./models/ud/vg_bas'
compos_model='./models/ud/vg_rc'
models = [base_model, compos_model]

transf_base_model='./models/transformed/vg_bas'
transf_compos_model='./models/transformed/vg_rc'
transformed_models = [transf_base_model, transf_compos_model]

datadir=None
transformed_datadir=None

#list of aux to filter out non aux
lemma_auxiliaries = \
        {
            'ca_ancora':
            [u'afirmar',
            u'anar',
            u'aprofitar',
            u'arribar',
            u'assegurar',
            u'començar',
            u'deixar',
            u'demanar',
            u'dir',
            u'estar',
            u'fer',
            u'fer-se',
            u'haver',
            u'intentar',
            u'poder',
            u'ser',
            u'tenir',
            u'tornar',
            u'voler'],
            'fi_tdt': [],
            'hr_set':[],
            'nl_alpino': []
        }
