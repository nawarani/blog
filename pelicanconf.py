#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os

AUTHOR = 'Anika Nawar'
SITENAME = 'blog'
SITEURL = 'https://nawarani.github.io/blog'
PATH = 'content'
STATIC_PATHS = ['images']
TIMEZONE = 'America/Toronto'
DEFAULT_LANG = 'en'
DEFAULT_PAGINATION = 10
# set to False for Production, True for Development
if os.environ.get('PELICAN_ENV') == 'DEV':
    RELATIVE_URLS = True
else:
    RELATIVE_URLS = False