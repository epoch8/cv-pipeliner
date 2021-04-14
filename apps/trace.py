import os
from contextlib import contextmanager

from opencensus.trace import config_integration, execution_context
from opencensus.trace.samplers import AlwaysOnSampler
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from opencensus.ext.flask.flask_middleware import FlaskMiddleware

config_integration.trace_integrations(['requests'])

def setup_flask_tracing(server):
    if os.environ.get('TRACING_ZIPKIN_HOST'):
        exporter = ZipkinExporter(service_name="cv-pipeliner",
                                    host_name=os.environ.get('TRACING_ZIPKIN_HOST'),
                                    port=9411,
                                    endpoint='/api/v2/spans')
    else:
        exporter = None

    middleware = FlaskMiddleware(
        server, 
        sampler=AlwaysOnSampler(),
        exporter=exporter,
        excludelist_paths=['_ah/health'])

def trace_function(f):
    def new_f(*args, **kwargs):
        with execution_context.get_opencensus_tracer().span(f.__name__):
            return f(*args, **kwargs)
    return new_f

@contextmanager
def trace_span(name):
    with execution_context.get_opencensus_tracer().span(name):
        yield 'ok'
