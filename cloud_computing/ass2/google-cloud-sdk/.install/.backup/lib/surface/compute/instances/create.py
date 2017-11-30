# Copyright 2014 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command for creating instances."""
import argparse
import re

from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import base_classes_resource_registry as resource_registry
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute import zone_utils
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.util import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log

DETAILED_HELP = {
    'DESCRIPTION': """\
        *{command}* facilitates the creation of Google Compute Engine
        virtual machines. For example, running:

          $ {command} example-instance-1 example-instance-2 example-instance-3 --zone us-central1-a

        will create three instances called `example-instance-1`,
        `example-instance-2`, and `example-instance-3` in the
        `us-central1-a` zone.

        When an instance is in RUNNING state and the system begins to boot,
        the instance creation is considered finished, and the command returns
        with a list of new virtual machines.  Note that you usually cannot log
        into a new instance until it finishes booting. Check the progress of an
        instance using `gcloud compute instances get-serial-port-output`.

        For more examples, refer to the *EXAMPLES* section below.
        """,
    'EXAMPLES': """\
        To create an instance with the latest ``Red Hat Enterprise Linux
        7'' image available, run:

          $ {command} example-instance --image-family rhel-7 --image-project rhel-cloud --zone us-central1-a
        """,
}


def _CommonArgs(parser,
                release_track,
                support_public_dns,
                support_public_ptr,
                support_network_tier,
                enable_regional=False,
                support_local_ssd_size=False,
                enable_kms=False):
  """Register parser args common to all tracks."""
  metadata_utils.AddMetadataArgs(parser)
  instances_flags.AddDiskArgs(parser, enable_regional, enable_kms=enable_kms)
  if release_track in [base.ReleaseTrack.ALPHA]:
    instances_flags.AddCreateDiskArgs(parser, enable_kms=enable_kms)
  if support_local_ssd_size:
    instances_flags.AddLocalSsdArgsWithSize(parser)
  else:
    instances_flags.AddLocalSsdArgs(parser)
  instances_flags.AddCanIpForwardArgs(parser)
  instances_flags.AddAddressArgs(
      parser, instances=True,
      support_network_tier=support_network_tier)
  instances_flags.AddAcceleratorArgs(parser)
  instances_flags.AddMachineTypeArgs(parser)
  deprecate_maintenance_policy = release_track in [base.ReleaseTrack.ALPHA]
  instances_flags.AddMaintenancePolicyArgs(parser, deprecate_maintenance_policy)
  instances_flags.AddNoRestartOnFailureArgs(parser)
  instances_flags.AddPreemptibleVmArgs(parser)
  instances_flags.AddServiceAccountAndScopeArgs(parser, False)
  instances_flags.AddTagsArgs(parser)
  instances_flags.AddCustomMachineTypeArgs(parser)
  instances_flags.AddNetworkArgs(parser)
  instances_flags.AddPrivateNetworkIpArgs(parser)
  instances_flags.AddImageArgs(parser)
  instances_flags.AddDeletionProtectionFlag(parser)
  if support_public_dns:
    instances_flags.AddPublicDnsArgs(parser, instance=True)
  if support_public_ptr:
    instances_flags.AddPublicPtrArgs(parser, instance=True)
  if support_network_tier:
    instances_flags.AddNetworkTierArgs(parser, instance=True)

  labels_util.AddCreateLabelsFlags(parser)
  instances_flags.AddMinCpuPlatformArgs(parser, release_track)

  parser.add_argument(
      '--description',
      help='Specifies a textual description of the instances.')

  instances_flags.INSTANCES_ARG_FOR_CREATE.AddArgument(
      parser, operation_type='create')

  csek_utils.AddCsekKeyArgs(parser)

  base.ASYNC_FLAG.AddToParser(parser)
  parser.display_info.AddFormat(
      resource_registry.RESOURCE_REGISTRY['compute.instances'].list_format)


@base.ReleaseTracks(base.ReleaseTrack.GA)
class Create(base.CreateCommand):
  """Create Google Compute Engine virtual machine instances."""

  _support_public_dns = False
  _support_public_ptr = False
  _support_network_tier = False

  @classmethod
  def Args(cls, parser):
    _CommonArgs(
        parser,
        release_track=base.ReleaseTrack.GA,
        support_public_dns=cls._support_public_dns,
        support_public_ptr=cls._support_public_ptr,
        support_network_tier=cls._support_network_tier)

  def Collection(self):
    return 'compute.instances'

  def GetSourceInstanceTemplate(self, args, resources):
    """Get sourceInstanceTemplate value as required by API."""
    return None

  def _CreateRequests(self, args, instance_refs,
                      compute_client, resource_parser):
    # gcloud creates default values for some fields in Instance resource
    # when no value was specified on command line.
    # When --source-instance-template was specified, defaults are taken from
    # Instance Template and gcloud flags are used to override them - by default
    # fields should not be initialized.
    source_instance_template = self.GetSourceInstanceTemplate(
        args, resource_parser)
    skip_defaults = source_instance_template is not None

    # This feature is only exposed in alpha/beta
    allow_rsa_encrypted = self.ReleaseTrack() in [base.ReleaseTrack.ALPHA,
                                                  base.ReleaseTrack.BETA]
    self.csek_keys = csek_utils.CsekKeyStore.FromArgs(args, allow_rsa_encrypted)

    if (skip_defaults and not args.IsSpecified('maintenance_policy') and
        not args.IsSpecified('preemptible') and
        not args.IsSpecified('restart_on_failure')):
      scheduling = None
    else:
      scheduling = instance_utils.CreateSchedulingMessage(
          messages=compute_client.messages,
          maintenance_policy=args.maintenance_policy,
          preemptible=args.preemptible,
          restart_on_failure=args.restart_on_failure)

    if args.tags:
      tags = compute_client.messages.Tags(items=args.tags)
    else:
      tags = None

    labels = None
    args_labels = getattr(args, 'labels', None)
    if args_labels:
      labels = compute_client.messages.Instance.LabelsValue(
          additionalProperties=[
              compute_client.messages.Instance.LabelsValue.AdditionalProperty(
                  key=key, value=value)
              for key, value in sorted(args.labels.iteritems())])

    if (skip_defaults and not args.IsSpecified('metadata') and
        not args.IsSpecified('metadata_from_file')):
      metadata = None
    else:
      metadata = metadata_utils.ConstructMetadataMessage(
          compute_client.messages,
          metadata=args.metadata,
          metadata_from_file=args.metadata_from_file)

    # If the user already provided an initial Windows password and
    # username through metadata, then there is no need to check
    # whether the image or the boot disk is Windows.

    boot_disk_size_gb = utils.BytesToGb(args.boot_disk_size)
    utils.WarnIfDiskSizeIsTooSmall(boot_disk_size_gb, args.boot_disk_type)

    # Check if the zone is deprecated or has maintenance coming.
    zone_resource_fetcher = zone_utils.ZoneResourceFetcher(compute_client)
    zone_resource_fetcher.WarnForZonalCreation(instance_refs)

    network_interface_arg = getattr(args, 'network_interface', None)
    if network_interface_arg:
      network_interfaces = instance_utils.CreateNetworkInterfaceMessages(
          resources=resource_parser,
          compute_client=compute_client,
          network_interface_arg=network_interface_arg,
          instance_refs=instance_refs,
          support_network_tier=self._support_network_tier)
    else:
      if self._support_public_dns is True:
        instances_flags.ValidatePublicDnsFlags(args)
      if self._support_public_ptr is True:
        instances_flags.ValidatePublicPtrFlags(args)

      if (skip_defaults and not args.IsSpecified('network') and
          not args.IsSpecified('subnet') and
          not args.IsSpecified('private_network_ip') and
          not args.IsSpecified('no_address') and
          not args.IsSpecified('address') and
          not args.IsSpecified('network_tier') and
          not args.IsSpecified('no_public_dns') and
          not args.IsSpecified('public_dns') and
          not args.IsSpecified('no_public_ptr') and
          not args.IsSpecified('public_ptr') and
          not args.IsSpecified('no_public_ptr_domain') and
          not args.IsSpecified('public_ptr_domain')):
        network_interfaces = []
      else:
        network_tier = getattr(args, 'network_tier', None)

        network_interfaces = [
            instance_utils.CreateNetworkInterfaceMessage(
                resources=resource_parser,
                compute_client=compute_client,
                network=args.network,
                subnet=args.subnet,
                private_network_ip=args.private_network_ip,
                no_address=args.no_address,
                address=args.address,
                instance_refs=instance_refs,
                network_tier=network_tier,
                no_public_dns=getattr(args, 'no_public_dns', None),
                public_dns=getattr(args, 'public_dns', None),
                no_public_ptr=getattr(args, 'no_public_ptr', None),
                public_ptr=getattr(args, 'public_ptr', None),
                no_public_ptr_domain=getattr(args, 'no_public_ptr_domain',
                                             None),
                public_ptr_domain=getattr(args, 'public_ptr_domain', None))
        ]

    if (skip_defaults and not args.IsSpecified('machine_type') and
        not args.IsSpecified('custom_cpu') and
        not args.IsSpecified('custom_memory')):
      machine_type_uris = [None for _ in instance_refs]
    else:
      machine_type_uris = instance_utils.CreateMachineTypeUris(
          resources=resource_parser,
          compute_client=compute_client,
          machine_type=args.machine_type,
          custom_cpu=args.custom_cpu,
          custom_memory=args.custom_memory,
          ext=getattr(args, 'custom_extensions', None),
          instance_refs=instance_refs)

    create_boot_disk = not instance_utils.UseExistingBootDisk(args.disk or [])
    if create_boot_disk:
      image_expander = image_utils.ImageExpander(compute_client,
                                                 resource_parser)
      image_uri, _ = image_expander.ExpandImageFlag(
          user_project=instance_refs[0].project,
          image=args.image,
          image_family=args.image_family,
          image_project=args.image_project,
          return_image_resource=False)
    else:
      image_uri = None

    # A list of lists where the element at index i contains a list of
    # disk messages that should be set for the instance at index i.
    disks_messages = []

    # A mapping of zone to boot disk references for all existing boot
    # disks that are being attached.
    # TODO(b/36050875): Simplify since resources.Resource is now hashable.
    existing_boot_disks = {}

    if (skip_defaults and not args.IsSpecified('disk') and
        not args.IsSpecified('create_disk') and
        not args.IsSpecified('local_ssd') and
        not args.IsSpecified('boot_disk_type') and
        not args.IsSpecified('boot_disk_device_name') and
        not args.IsSpecified('boot_disk_auto_delete') and
        not args.IsSpecified('boot_disk_kms_key') and
        not args.IsSpecified('boot_disk_kms_project') and
        not args.IsSpecified('boot_disk_kms_location') and
        not args.IsSpecified('boot_disk_kms_keyring') and
        not args.IsSpecified('require_csek_key_create')):
      disks_messages = [[] for _ in instance_refs]
    else:
      for instance_ref in instance_refs:
        persistent_disks, boot_disk_ref = (
            instance_utils.CreatePersistentAttachedDiskMessages(
                resource_parser, compute_client, self.csek_keys,
                args.disk or [], instance_ref))
        persistent_create_disks = (
            instance_utils.CreatePersistentCreateDiskMessages(
                compute_client,
                resource_parser,
                self.csek_keys,
                getattr(args, 'create_disk', []),
                instance_ref))
        local_ssds = []
        for x in args.local_ssd or []:
          local_ssds.append(
              instance_utils.CreateLocalSsdMessage(
                  resource_parser,
                  compute_client.messages,
                  x.get('device-name'),
                  x.get('interface'),
                  x.get('size'),
                  instance_ref.zone,
                  instance_ref.project)
          )

        if create_boot_disk:
          boot_disk = instance_utils.CreateDefaultBootAttachedDiskMessage(
              compute_client, resource_parser,
              disk_type=args.boot_disk_type,
              disk_device_name=args.boot_disk_device_name,
              disk_auto_delete=args.boot_disk_auto_delete,
              disk_size_gb=boot_disk_size_gb,
              require_csek_key_create=(
                  args.require_csek_key_create if self.csek_keys else None),
              image_uri=image_uri,
              instance_ref=instance_ref,
              csek_keys=self.csek_keys,
              kms_args=args)
          persistent_disks = [boot_disk] + persistent_disks
        else:
          existing_boot_disks[boot_disk_ref.zone] = boot_disk_ref
        disks_messages.append(persistent_disks + persistent_create_disks +
                              local_ssds)

    accelerator_args = getattr(args, 'accelerator', None)

    project_to_sa = {}
    requests = []
    for instance_ref, machine_type_uri, disks in zip(
        instance_refs, machine_type_uris, disks_messages):
      if instance_ref.project not in project_to_sa:
        scopes = None
        if not args.no_scopes and not args.scopes:
          # User didn't provide any input on scopes. If project has no default
          # service account then we want to create a VM with no scopes
          request = (compute_client.apitools_client.projects,
                     'Get',
                     compute_client.messages.ComputeProjectsGetRequest(
                         project=instance_ref.project))
          errors = []
          result = compute_client.MakeRequests([request], errors)
          if not errors:
            if not result[0].defaultServiceAccount:
              scopes = []
              log.status.Print(
                  'There is no default service account for project {}. '
                  'Instance {} will not have scopes.'.format(
                      instance_ref.project, instance_ref.Name))
        if scopes is None:
          scopes = [] if args.no_scopes else args.scopes

        if args.no_service_account:
          service_account = None
        else:
          service_account = args.service_account
        if (skip_defaults and not args.IsSpecified('scopes') and
            not args.IsSpecified('no_scopes') and
            not args.IsSpecified('service_account') and
            not args.IsSpecified('no_service_account')):
          service_accounts = []
        else:
          service_accounts = instance_utils.CreateServiceAccountMessages(
              messages=compute_client.messages,
              scopes=scopes,
              service_account=service_account)
        project_to_sa[instance_ref.project] = service_accounts

      if skip_defaults and not args.IsSpecified('can_ip_forward'):
        can_ip_forward = None
      else:
        can_ip_forward = args.can_ip_forward

      instance = compute_client.messages.Instance(
          canIpForward=can_ip_forward,
          disks=disks,
          description=args.description,
          machineType=machine_type_uri,
          metadata=metadata,
          minCpuPlatform=args.min_cpu_platform,
          name=instance_ref.Name(),
          networkInterfaces=network_interfaces,
          serviceAccounts=project_to_sa[instance_ref.project],
          scheduling=scheduling,
          tags=tags)
      if getattr(args, 'deletion_protection', None) is not None:
        instance.deletionProtection = args.deletion_protection
      if labels:
        instance.labels = labels
      if accelerator_args:
        accelerator_type_name = accelerator_args['type']
        accelerator_type_ref = resource_parser.Parse(
            accelerator_type_name,
            collection='compute.acceleratorTypes',
            params={'project': instance_ref.project,
                    'zone': instance_ref.zone})
        # Accelerator count is default to 1.
        accelerator_count = int(accelerator_args.get('count', 1))
        accelerators = instance_utils.CreateAcceleratorConfigMessages(
            compute_client.messages, accelerator_type_ref,
            accelerator_count)
        instance.guestAccelerators = accelerators

      request = compute_client.messages.ComputeInstancesInsertRequest(
          instance=instance,
          project=instance_ref.project,
          zone=instance_ref.zone)

      if source_instance_template:
        request.sourceInstanceTemplate = source_instance_template

      sole_tenancy_host_arg = getattr(args, 'sole_tenancy_host', None)
      if sole_tenancy_host_arg:
        sole_tenancy_host_ref = resource_parser.Parse(
            sole_tenancy_host_arg, collection='compute.hosts',
            params={
                'project': instance_ref.project,
                'zone': instance_ref.zone
            })
        request.instance.host = sole_tenancy_host_ref.SelfLink()
      requests.append(
          (compute_client.apitools_client.instances, 'Insert', request))
    return requests

  def Run(self, args):
    instances_flags.ValidateDiskFlags(
        args, enable_kms=self.ReleaseTrack() in [base.ReleaseTrack.ALPHA])
    instances_flags.ValidateLocalSsdFlags(args)
    instances_flags.ValidateNicFlags(args)
    instances_flags.ValidateServiceAccountAndScopeArgs(args)
    instances_flags.ValidateAcceleratorArgs(args)
    if self._support_network_tier:
      instances_flags.ValidateNetworkTierArgs(args)

    holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
    compute_client = holder.client
    resource_parser = holder.resources

    instance_refs = instances_flags.INSTANCES_ARG.ResolveAsResource(
        args, resource_parser,
        scope_lister=flags.GetDefaultScopeLister(compute_client))

    requests = self._CreateRequests(args, instance_refs,
                                    compute_client, resource_parser)

    if not args.async:
      # TODO(b/63664449): Replace this with poller + progress tracker.
      try:
        # Using legacy MakeRequests (which also does polling) here until
        # replaced by api_lib.utils.waiter.
        return compute_client.MakeRequests(requests)
      except exceptions.ToolException as e:
        invalid_machine_type_message_regex = (
            r'Invalid value for field \'resource.machineType\': .+. '
            r'Machine type with name \'.+\' does not exist in zone \'.+\'\.')
        if re.search(invalid_machine_type_message_regex, e.message):
          raise exceptions.ToolException(
              e.message +
              '\nUse `gcloud compute machine-types list --zones` to see the '
              'available machine  types.')
        raise

    errors_to_collect = []
    responses = compute_client.BatchRequests(requests, errors_to_collect)
    for r in responses:
      err = getattr(r, 'error', None)
      if err:
        errors_to_collect.append(poller.OperationErrors(err.errors))
    if errors_to_collect:
      raise core_exceptions.MultiError(errors_to_collect)

    operation_refs = [holder.resources.Parse(r.selfLink) for r in responses]

    for instance_ref, operation_ref in zip(instance_refs, operation_refs):
      log.status.Print('Instance creation in progress for [{}]: {}'
                       .format(instance_ref.instance, operation_ref.SelfLink()))
    log.status.Print('Use [gcloud compute operations describe URI] command '
                     'to check the status of the operation(s).')
    if not args.IsSpecified('format'):
      # For async output we need a separate format. Since we already printed in
      # the status messages information about operations there is nothing else
      # needs to be printed.
      args.format = 'disable'
    return responses


@base.ReleaseTracks(base.ReleaseTrack.BETA)
class CreateBeta(Create):
  """Create Google Compute Engine virtual machine instances."""

  _support_public_dns = False
  _support_public_ptr = True
  _support_network_tier = False

  @classmethod
  def Args(cls, parser):
    _CommonArgs(
        parser,
        release_track=base.ReleaseTrack.BETA,
        support_public_dns=cls._support_public_dns,
        support_public_ptr=cls._support_public_ptr,
        support_network_tier=cls._support_network_tier)


@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class CreateAlpha(Create):
  """Create Google Compute Engine virtual machine instances."""

  _support_public_dns = True
  _support_public_ptr = True
  _support_network_tier = True

  @classmethod
  def Args(cls, parser):
    parser.add_argument('--sole-tenancy-host', help=argparse.SUPPRESS)
    _CommonArgs(
        parser,
        release_track=base.ReleaseTrack.ALPHA,
        support_public_dns=cls._support_public_dns,
        support_public_ptr=cls._support_public_ptr,
        support_network_tier=cls._support_network_tier,
        enable_regional=True,
        support_local_ssd_size=True,
        enable_kms=True)
    CreateAlpha.SOURCE_INSTANCE_TEMPLATE = (
        instances_flags.MakeSourceInstanceTemplateArg())
    CreateAlpha.SOURCE_INSTANCE_TEMPLATE.AddArgument(parser)

  def GetSourceInstanceTemplate(self, args, resources):
    if not args.IsSpecified('source_instance_template'):
      return None
    ref = self.SOURCE_INSTANCE_TEMPLATE.ResolveAsResource(args, resources)
    return ref.SelfLink()


Create.detailed_help = DETAILED_HELP
