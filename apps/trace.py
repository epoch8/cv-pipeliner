from opencensus.trace.tracer import Tracer
from opencensus.trace.samplers import AlwaysOnSampler

# Initialize a tracer, by default using the `PrintExporter`
tracer = Tracer(sampler=AlwaysOnSampler())
