c = get_config()

#Export all the notebooks in the current directory to the sphinx_howto format.
c.NbConvertApp.export_format = 'html'

c.NbConvertApp.use_output_suffix = False
c.NbConvertApp.notebooks = ['notebooks/*.ipynb']
c.FilesWriter.build_directory = '_posts/notebooks-html/'
c.TemplateExporter.template_file = 'ipynb_to_md.tpl'
